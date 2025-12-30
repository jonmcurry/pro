fn main() {
    // Embed Windows manifest for Common Controls v6
    // This is required for NWG to work correctly
    embed_resource::compile("windows-manifest.rc", embed_resource::NONE);
}
