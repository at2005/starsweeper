mod query;
mod tap;
use anyhow::Result;
use tap::DownloadManager;

#[tokio::main]
async fn main() -> Result<()> {
    let url = "https://gea.esac.esa.int";
    let mut download_manager = DownloadManager::new(url);
    download_manager.start_all().await.unwrap();
    Ok(())
}
