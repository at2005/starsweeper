use crate::query::get_possible_orbits;
use anyhow::Result;
use futures_util::{future::join_all, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use roxmltree::Document;
use std::fs::File;
use std::io::Write;
use url::form_urlencoded;

fn format_request(query: String) -> String {
    let encoded: String = form_urlencoded::Serializer::new(String::new())
        .append_pair("REQUEST", "doQuery")
        .append_pair("LANG", "ADQL")
        .append_pair("FORMAT", "votable_plain")
        .append_pair("PHASE", "RUN")
        .append_pair("QUERY", &query)
        .finish();

    encoded
}

async fn make_request(
    tap_client: &TapClient,
    method: reqwest::Method,
    path: &str,
    query: String,
) -> reqwest::Response {
    let client_builder = tap_client.client.request(
        method,
        format!("{}:{}{}", tap_client.url, tap_client.port, path),
    );
    let response = match query {
        query if query.is_empty() => client_builder.send().await.unwrap(),
        _ => client_builder
            .body(format_request(query))
            .send()
            .await
            .unwrap(),
    };

    response
}

pub struct TapClient {
    url: String,
    client: reqwest::Client,
    port: u16,
}

impl Clone for TapClient {
    fn clone(&self) -> Self {
        TapClient {
            url: self.url.clone(),
            client: self.client.clone(),
            port: self.port,
        }
    }
}

impl TapClient {
    fn new(url: String) -> TapClient {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/x-www-form-urlencoded"),
        );

        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("text/plain"),
        );

        TapClient {
            url: url,
            client: reqwest::Client::builder()
                .default_headers(headers.clone())
                .build()
                .unwrap(),
            port: 443,
        }
    }

    pub async fn start_download_job(&self, progress_bar: ProgressBar, query: String) -> Result<()> {
        let tables_path: &str = "/tap-server/tap/async";
        let response = make_request(&self, reqwest::Method::POST, tables_path, query).await;
        let response_text = response.text().await.unwrap();
        let xml_doc = Document::parse(&response_text).unwrap();
        let job_id = xml_doc
            .descendants()
            .find(|n| {
                n.tag_name().name() == "jobId"
                    && n.tag_name().namespace() == Some("http://www.ivoa.net/xml/UWS/v1.0")
            })
            .and_then(|n| n.text())
            .unwrap();

        loop {
            let status_response = make_request(
                &self,
                reqwest::Method::GET,
                &format!("{tables_path}/{job_id}"),
                "".to_string(),
            )
            .await
            .text()
            .await
            .unwrap();
            let xml_doc = Document::parse(&status_response).unwrap();
            let phase = xml_doc
                .descendants()
                .find(|n| {
                    n.tag_name().name() == "phase"
                        && n.tag_name().namespace() == Some("http://www.ivoa.net/xml/UWS/v1.0")
                })
                .and_then(|n| n.text())
                .unwrap();

            if phase == "COMPLETED" {
                break;
            }

            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        let tables_path_clone = tables_path.to_string();
        let resource_path = format!("{}/{}", tables_path_clone, job_id);
        let result = make_request(
            &self,
            reqwest::Method::GET,
            &format!("{}/results/result", resource_path),
            "".to_string(),
        )
        .await;

        assert!(result.status().is_success());

        let output_filename = format!("data/{}.vot", job_id);
        let mut output_file = File::create(output_filename).unwrap();
        progress_bar.set_length(5 * 1024 * 1024);
        let mut stream = result.bytes_stream();
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    output_file.write_all(&chunk).unwrap();
                    progress_bar.inc(chunk.len() as u64);
                }
                Err(_e) => {
                    break;
                }
            }
        }
        Ok(())
    }
}

pub struct DownloadManager {
    tap_client: TapClient,
    multi_progress: MultiProgress,
}

impl DownloadManager {
    pub fn new(url: &str) -> DownloadManager {
        let tap_client = TapClient::new(url.to_string());
        let multi_progress = MultiProgress::new();
        DownloadManager {
            tap_client,
            multi_progress,
        }
    }

    pub async fn start_all(&mut self) -> Result<()> {
        let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?;
        std::fs::create_dir_all("data")?;
        let possible_orbits = get_possible_orbits();
        let mut handles = vec![];
        let num_jobs = 5;
        for i in 0..num_jobs {
            let query = possible_orbits[i].get_query();
            let tap_client = self.tap_client.clone();
            let progress_bar = self.multi_progress.add(ProgressBar::new(0));
            progress_bar.set_style(style.clone());
            println!("Starting job {}", i + 1);
            let handle = tokio::spawn(async move {
                tap_client
                    .start_download_job(progress_bar, query)
                    .await
                    .unwrap();
            });
            handles.push(handle);
        }

        join_all(handles).await;

        Ok(())
    }
}
