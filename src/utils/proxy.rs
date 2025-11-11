use std::env;

pub struct ProxyGuard;

impl ProxyGuard {
    pub fn new(port: u16) -> Self {
        unsafe {
            env::set_var("HTTPS_PROXY", format!("http://127.0.0.1:{port}"));
        }
        Self
    }
}

impl Drop for ProxyGuard {
    fn drop(&mut self) {
        unsafe {
            env::remove_var("HTTPS_PROXY");
        }
    }
}
