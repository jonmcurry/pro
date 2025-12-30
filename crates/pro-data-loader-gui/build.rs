fn main() {
    // Embed Windows manifest to declare dependency on Common Controls v6
    // This is required for native-windows-gui to work properly with windows-sys
    // See: https://github.com/gabdube/native-windows-gui/issues/251
    #[cfg(windows)]
    {
        embed_resource::compile("pro-data-loader-gui-manifest.rc", embed_resource::NONE);
    }
}
