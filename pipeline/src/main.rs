use std::io::Write;

use anyhow::Result;
use futures_util::{future::join_all, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest;
use scraper::{Html, Selector};
use tokio;
use tokio::task::JoinHandle;

struct Downloader {
    url: String,
    filename: String,
    client: reqwest::Client,
    progress_bar: ProgressBar,
}

impl Downloader {
    fn new(url: String, filename: String, progress_bar: ProgressBar) -> Downloader {
        Downloader {
            url: url,
            filename: filename,
            client: reqwest::Client::new(),
            progress_bar: progress_bar,
        }
    }

    async fn start_download(&self) -> (JoinHandle<()>, JoinHandle<()>) {
        let notify = std::sync::Arc::new(tokio::sync::Notify::new());
        let notify2 = notify.clone();
        let filename = self.filename.clone();
        let filename2 = filename.clone();
        let url = self.url.clone();
        let client = self.client.clone();
        let progress_bar = self.progress_bar.clone();
        
        let handle1 = tokio::spawn(async move {
            let response = client.get(&url).send().await.unwrap();
            let total_size = response.content_length().unwrap();
            progress_bar.set_length(total_size);

            let mut file = std::fs::File::create(format!("data/{}", &filename)).unwrap();

            let mut downloaded = 0;
            let mut stream = response.bytes_stream();
            while let Some(item) = stream.next().await {
                let chunk = item.unwrap();
                file.write_all(&chunk).unwrap();
                let _ = progress_bar.inc(chunk.len() as u64);
                downloaded += chunk.len();
                progress_bar.set_position(downloaded as u64);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }

            progress_bar.finish_with_message(format!("Downloaded {}", &filename));
            notify.notify_one();
        });

        let handle2 = tokio::spawn(async move {
            notify2.notified().await;
            let status = tokio::process::Command::new("gzip")
                .arg("-d")
                .arg(&filename2)
                .spawn()
                .unwrap()
                .wait()
                .await
                .unwrap();

            if status.success() {
                println!("Unzipped {}", &filename2);
            } else {
                println!("Failed to unzip {}", &filename2);
            }
        });

        (handle1, handle2)
    }
}

struct DownloadMgr {
    base_url: String,
}

impl DownloadMgr {
    fn new(url: &str) -> DownloadMgr {
        DownloadMgr {
            base_url: url.to_string(),
        }
    }

    async fn get_file_urls(&mut self) -> Result<Vec<String>> {
        let response = reqwest::get(&self.base_url).await?;
        let html = response.text().await?;
        let document = Html::parse_document(&html);
        let a_selector = Selector::parse("a").unwrap();
        let urls = document
            .select(&a_selector)
            .map(|node| {
                let href = node.value().attr("href").unwrap();
                format!("{}{}", self.base_url, href)
            })
            .filter(|url| url.ends_with(".gz"));

        Ok(urls.collect())
    }

    async fn start_all(&mut self) -> Result<()> {
        let multi_progress = MultiProgress::new();
        let urls = self.get_file_urls().await?;
        let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?;
        std::fs::create_dir_all("data")?;

        let mut handles: Vec<JoinHandle<()>> = vec![];
        for url in urls {
            let filename = url.split("/").last().unwrap().to_string();
            let progress_bar = multi_progress.add(ProgressBar::new(0));
            progress_bar.set_style(style.clone());
            let downloader = Downloader::new(url, filename, progress_bar);
            let (handle1, handle2) = downloader.start_download().await;
            handles.push(handle1);
            handles.push(handle2);
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        }

        join_all(handles).await;

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let url = "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/";
    let mut download_manager = DownloadMgr::new(url);
    download_manager.start_all().await.unwrap();
    Ok(())
}
