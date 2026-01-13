fn main() {
    // Embed Windows manifest for Common Controls v6
    embed_resource::compile("windows-manifest.rc", embed_resource::NONE);
}
