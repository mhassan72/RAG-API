mod server;
mod grpc;
mod ml;
mod search;
mod cache;
mod database;
mod error;
mod types;
mod config;

use crate::server::SearchServer;
use crate::error::SearchError;
use crate::config::Config;

#[tokio::main]
async fn main() -> Result<(), SearchError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .json()
        .init();

    tracing::info!("Starting RAG Search API server");

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!("Configuration loaded successfully");
    tracing::info!("Server will listen on {}:{}", config.server.host, config.server.port);

    let server = SearchServer::new(config).await?;
    
    // For now, just run the HTTP server
    // gRPC functionality is available via the GrpcSearchService
    tracing::info!("Starting HTTP server (gRPC service available programmatically)");
    server.run().await?;

    Ok(())
}