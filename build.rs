fn main() {
    println!("cargo:rerun-if-env-changed=OMNI_TUNED_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=NEEDLE_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_BUNDLED_NEEDLE").is_some() {
        let dir = std::env::var_os("NEEDLE_LIB_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(".cache/needle"));
        assert!(
            dir.join("libneedle.so").is_file(),
            "Needle library missing. Run python3 scripts/setup_model.py, or build --no-default-features for engine-only development."
        );
        println!(
            "cargo:rerun-if-changed={}",
            dir.join("libneedle.so").display()
        );
        println!(
            "cargo:rustc-env=OMNI_NEEDLE_LIBRARY={}",
            std::fs::canonicalize(dir.join("libneedle.so"))
                .unwrap()
                .display()
        );
    }
    if std::env::var_os("CARGO_FEATURE_TUNED_MODEL").is_some() {
        let weights = std::env::var_os("OMNI_TUNED_MODEL_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("models/omni.cact"));
        assert!(
            weights.is_file(),
            "Tuned weights missing: supply models/omni.cact or OMNI_TUNED_MODEL_PATH."
        );
        println!("cargo:rerun-if-changed={}", weights.display());
        println!(
            "cargo:rustc-env=OMNI_TUNED_WEIGHTS={}",
            std::fs::canonicalize(weights).unwrap().display()
        );
    }
}
