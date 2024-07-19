#[macro_use]
extern crate tracing;

use std::io::Write;

use anyhow::Result;
use candle_core::{cuda, quantized};
use candle_core::utils::{with_avx, with_f16c, with_neon, with_simd128};

#[tokio::main]
async fn main() -> Result<()> {
    // tracing_subscriber::fmt::init();

    quantized::cuda::set_force_dmmv(false);

    cuda::set_gemm_reduced_precision_f16(true);
    cuda::set_gemm_reduced_precision_bf16(true);

    info!( "avx: {}, neon: {}, simd128: {}, f16c: {}",
            with_avx(),
            with_neon(),
            with_simd128(),
            with_f16c()
        );

    let mut bot = openchat_bot::ChatBot::from_default_args().await?;

    loop {
        print!("> ");
        // 把缓冲区字符串刷到控制台上
        std::io::stdout().flush()?;
        let mut prompt = String::new();
        std::io::stdin().read_line(&mut prompt)?;

        print!("bot: ");

        let mut rx = bot.chat(prompt);

        while let Some(word) = rx.recv().await {
            print!("{word}");
            std::io::stdout().flush()?;
        }

        println!();
    }

}
