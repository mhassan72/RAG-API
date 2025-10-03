use rag_search_api::TokenizerService;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RAG Search API - TokenizerService Demo ===\n");
    
    // Create a new tokenizer service
    let tokenizer = TokenizerService::new()?;
    
    // Demo 1: Query normalization
    println!("1. Query Normalization:");
    let messy_query = "  Hello    WORLD!\t\nHow are YOU?  ";
    let normalized = tokenizer.normalize_query(messy_query);
    println!("   Original: {:?}", messy_query);
    println!("   Normalized: {:?}", normalized);
    println!();
    
    // Demo 2: Text cleaning
    println!("2. Text Cleaning:");
    let messy_text = "Hello@#$%World!!! This~is*a&test(){}[]|\\+=<>?/";
    let cleaned = tokenizer.clean_text(messy_text);
    println!("   Original: {:?}", messy_text);
    println!("   Cleaned: {:?}", cleaned);
    println!();
    
    // Demo 3: Cache key generation
    println!("3. Cache Key Generation:");
    let queries = vec![
        "machine learning algorithms",
        "  Machine Learning   ALGORITHMS  ",
        "MACHINE\tLEARNING\nALGORITHMS",
    ];
    
    for query in &queries {
        let cache_key = tokenizer.generate_cache_key(query);
        println!("   Query: {:?} -> Cache Key: {}", query, cache_key);
    }
    println!("   (Note: All three queries produce the same cache key due to normalization)");
    println!();
    
    // Demo 4: Cache key with parameters
    println!("4. Cache Key with Parameters:");
    let base_query = "artificial intelligence";
    let mut filters = HashMap::new();
    filters.insert("language".to_string(), "en".to_string());
    filters.insert("frozen".to_string(), "false".to_string());
    
    let scenarios = vec![
        (10, None, "Basic search (k=10)"),
        (20, None, "More results (k=20)"),
        (10, Some(0.5), "With min_score=0.5"),
        (10, Some(0.8), "With min_score=0.8"),
    ];
    
    for (k, min_score, description) in scenarios {
        let cache_key = tokenizer.generate_cache_key_with_params(
            base_query, 
            k, 
            min_score, 
            &filters
        );
        println!("   {}: {}", description, cache_key);
    }
    println!();
    
    // Demo 5: Query validation
    println!("5. Query Validation:");
    let long_query_1000 = "a".repeat(1000);
    let long_query_1001 = "a".repeat(1001);
    
    let test_queries = vec![
        ("valid query", true),
        ("", false),
        ("   ", false),
        ("!@#$%^&*()", false),
        (long_query_1000.as_str(), true),
        (long_query_1001.as_str(), false),
        ("hello world", true),
        ("123 test", true),
    ];
    
    for (query, should_be_valid) in test_queries {
        let is_valid = tokenizer.validate_query(query).is_ok();
        let status = if is_valid { "✓ VALID" } else { "✗ INVALID" };
        let expected = if should_be_valid { "✓" } else { "✗" };
        
        let display_query = if query.len() > 50 { 
            format!("{}...", &query[..47])
        } else { 
            query.to_string()
        };
        
        println!("   {} Query: {:?} -> {} (expected: {})", 
                 expected, 
                 display_query,
                 status,
                 if should_be_valid { "valid" } else { "invalid" });
    }
    println!();
    
    // Demo 6: Edge cases
    println!("6. Edge Cases:");
    let edge_cases = vec![
        ("Unicode: café résumé naïve", "café résumé naïve"),
        ("Emojis: hello 😀 world", "hello 😀 world"),
        ("Mixed case: Hello, WORLD!", "hello, world!"),
        ("Control chars: hello\x00world\x7f", "helloworld"),
    ];
    
    for (description, input) in edge_cases {
        let normalized = tokenizer.normalize_query(input);
        println!("   {}", description);
        println!("     Input: {:?}", input);
        println!("     Output: {:?}", normalized);
    }
    
    println!("\n=== Demo Complete ===");
    Ok(())
}