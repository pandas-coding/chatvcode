use std::path::Path;

/// Default directory names that should be ignored during scanning.
const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".svn",
    ".hg",
    "target",
    "node_modules",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
];

/// Checks if a directory should be ignored based on its name.
///
/// This function implements the standard ignore rules for common
/// directories that typically don't contain user source code.
#[must_use]
pub fn should_ignore_dir(path: &Path) -> bool {
    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
        IGNORED_DIRS.contains(&dir_name)
    } else {
        false
    }
}

/// Checks if a file should be ignored based on its name/extension.
///
/// Filters out common non-source files like binaries, images, etc.
#[must_use]
pub fn should_ignore_file(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
        return true;
    };

    // Skip hidden files (starting with dot)
    if file_name.starts_with('.') {
        return true;
    }

    // Skip common non-source extensions
    let ignored_extensions = [
        // Binary/executable
        "exe", "dll", "so", "dylib", "o", "a", "lib", // Images
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "webp", // Audio/Video
        "mp3", "mp4", "wav", "avi", "mov", // Documents
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", // Archives
        "zip", "tar", "gz", "rar", "7z", // Compiled files
        "pyc", "pyo", "class", "beam", // Lock files and other generated files
        "lock",
    ];

    if let Some(ext) = path.extension().and_then(|e| e.to_str())
        && ignored_extensions.contains(&ext.to_lowercase().as_str())
    {
        return true;
    }

    // Skip specific files
    let ignored_files = [
        ".DS_Store",
        "Thumbs.db",
        "Cargo.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    ];

    if ignored_files.contains(&file_name) {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_ignore_git_dir() {
        assert!(should_ignore_dir(Path::new("/project/.git")));
        assert!(should_ignore_dir(Path::new(".git")));
    }

    #[test]
    fn test_should_ignore_target_dir() {
        assert!(should_ignore_dir(Path::new("/project/target")));
        assert!(should_ignore_dir(Path::new("target")));
    }

    #[test]
    fn test_should_ignore_node_modules() {
        assert!(should_ignore_dir(Path::new("/project/node_modules")));
        assert!(should_ignore_dir(Path::new("node_modules")));
    }

    #[test]
    fn test_should_not_ignore_normal_dirs() {
        assert!(!should_ignore_dir(Path::new("/project/src")));
        assert!(!should_ignore_dir(Path::new("crates")));
        assert!(!should_ignore_dir(Path::new("tests")));
    }

    #[test]
    fn test_should_ignore_binary_files() {
        assert!(should_ignore_file(Path::new("program.exe")));
        assert!(should_ignore_file(Path::new("lib.so")));
        assert!(should_ignore_file(Path::new("lib.dylib")));
    }

    #[test]
    fn test_should_ignore_image_files() {
        assert!(should_ignore_file(Path::new("image.png")));
        assert!(should_ignore_file(Path::new("photo.jpg")));
        assert!(should_ignore_file(Path::new("icon.svg")));
    }

    #[test]
    fn test_should_ignore_hidden_files() {
        assert!(should_ignore_file(Path::new(".gitignore")));
        assert!(should_ignore_file(Path::new(".env")));
    }

    #[test]
    fn test_should_ignore_lock_files() {
        assert!(should_ignore_file(Path::new("Cargo.lock")));
        assert!(should_ignore_file(Path::new("package-lock.json")));
        assert!(should_ignore_file(Path::new("yarn.lock")));
    }

    #[test]
    fn test_should_not_ignore_source_files() {
        assert!(!should_ignore_file(Path::new("main.rs")));
        assert!(!should_ignore_file(Path::new("index.js")));
        assert!(!should_ignore_file(Path::new("app.tsx")));
        assert!(!should_ignore_file(Path::new("lib.rs")));
    }

    #[test]
    fn test_should_ignore_svn_dir() {
        assert!(should_ignore_dir(Path::new("/project/.svn")));
        assert!(should_ignore_dir(Path::new(".svn")));
    }

    #[test]
    fn test_should_ignore_hg_dir() {
        assert!(should_ignore_dir(Path::new("/project/.hg")));
        assert!(should_ignore_dir(Path::new(".hg")));
    }

    #[test]
    fn test_should_ignore_dist_dir() {
        assert!(should_ignore_dir(Path::new("dist")));
        assert!(should_ignore_dir(Path::new("/project/dist")));
    }

    #[test]
    fn test_should_ignore_build_dir() {
        assert!(should_ignore_dir(Path::new("build")));
        assert!(should_ignore_dir(Path::new("/project/build")));
    }

    #[test]
    fn test_should_ignore_pycache_dir() {
        assert!(should_ignore_dir(Path::new("__pycache__")));
        assert!(should_ignore_dir(Path::new("/project/__pycache__")));
    }

    #[test]
    fn test_should_ignore_venv_dirs() {
        assert!(should_ignore_dir(Path::new(".venv")));
        assert!(should_ignore_dir(Path::new("venv")));
    }

    #[test]
    fn test_should_ignore_next_dirs() {
        assert!(should_ignore_dir(Path::new(".next")));
        assert!(should_ignore_dir(Path::new(".nuxt")));
    }

    #[test]
    fn test_should_ignore_audio_video_files() {
        assert!(should_ignore_file(Path::new("audio.mp3")));
        assert!(should_ignore_file(Path::new("video.mp4")));
        assert!(should_ignore_file(Path::new("sound.wav")));
    }

    #[test]
    fn test_should_ignore_document_files() {
        assert!(should_ignore_file(Path::new("doc.pdf")));
        assert!(should_ignore_file(Path::new("sheet.xlsx")));
        assert!(should_ignore_file(Path::new("presentation.pptx")));
    }

    #[test]
    fn test_should_ignore_archive_files() {
        assert!(should_ignore_file(Path::new("archive.zip")));
        assert!(should_ignore_file(Path::new("backup.tar")));
        assert!(should_ignore_file(Path::new("data.gz")));
        assert!(should_ignore_file(Path::new("pack.7z")));
    }

    #[test]
    fn test_should_ignore_compiled_files() {
        assert!(should_ignore_file(Path::new("module.pyc")));
        assert!(should_ignore_file(Path::new("Main.class")));
    }

    #[test]
    fn test_should_ignore_ds_store() {
        assert!(should_ignore_file(Path::new(".DS_Store")));
    }

    #[test]
    fn test_should_ignore_thumbs_db() {
        assert!(should_ignore_file(Path::new("Thumbs.db")));
    }

    #[test]
    fn test_should_ignore_pnpm_lock() {
        assert!(should_ignore_file(Path::new("pnpm-lock.yaml")));
    }

    #[test]
    fn test_should_not_ignore_ts_files() {
        assert!(!should_ignore_file(Path::new("app.ts")));
        assert!(!should_ignore_file(Path::new("component.tsx")));
    }

    #[test]
    fn test_should_not_ignore_rust_files() {
        assert!(!should_ignore_file(Path::new("main.rs")));
        assert!(!should_ignore_file(Path::new("lib.rs")));
    }

    #[test]
    fn test_should_not_ignore_js_files() {
        assert!(!should_ignore_file(Path::new("index.js")));
        assert!(!should_ignore_file(Path::new("component.jsx")));
    }

    #[test]
    fn test_should_ignore_file_with_no_name() {
        let path = Path::new("");
        assert!(should_ignore_file(path));
    }

    #[test]
    fn test_should_ignore_binary_extensions_case_insensitive() {
        assert!(should_ignore_file(Path::new("program.EXE")));
        assert!(should_ignore_file(Path::new("image.PNG")));
        assert!(should_ignore_file(Path::new("archive.ZIP")));
    }
}
