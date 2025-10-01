mod server;
mod ml;
mod search;
mod cache;
mod error;
mod types;

use crate::server::SearchServer;
use crate::error::SearchError;

#[tokio::main]
async fn main() -> Result<(), SearchError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .json()
        .init();

    tracing::info!("Starting RAG Search API server");

    let server = SearchServer::new().await?;
    server.run().await?;

    Ok(())
}