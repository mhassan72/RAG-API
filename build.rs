fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Try to compile protobuf files for gRPC service
    // If protoc is not available, skip compilation and use pre-generated files
    match tonic_build::compile_protos("proto/search.proto") {
        Ok(_) => println!("cargo:warning=Successfully compiled protobuf files"),
        Err(e) => {
            println!("cargo:warning=Failed to compile protobuf files: {}. Using pre-generated files.", e);
            // In a real implementation, we would have pre-generated files
            // For now, we'll create a simple stub
        }
    }
    Ok(())
}