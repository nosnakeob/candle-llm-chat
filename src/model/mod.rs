use anyhow::Result;
use candle::Tensor;
use candle_transformers::models::{quantized_qwen3, quantized_qwen3_moe, qwen3};
pub mod config;
pub mod hub;
pub mod qwen3_5;
pub mod qwen3_vl;
pub mod registry;

/// 视觉模型的配置信息（非视觉模型返回 None）
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_token_id: u32,
}

/// 视觉输入数据，传递给 forward 的 vision 参数
pub struct VisionInput {
    pub pixel_values: Tensor,
    pub image_grid_thw: Tensor,
    pub image_token_span: Vec<Vec<(usize, usize)>>,
}

macro_rules! impl_model_traits {
    ($($model:ty),+ $(,)?) => {
        $(
            impl crate::model::ModelInference for $model {
                fn forward(
                    &mut self,
                    x: &candle::Tensor,
                    index_pos: usize,
                    _vision: Option<&crate::model::VisionInput>,
                ) -> anyhow::Result<candle::Tensor> {
                    self.forward(x, index_pos).map_err(anyhow::Error::msg)
                }

                fn reset_kv_cache(&mut self) -> anyhow::Result<()> {
                    // 显式走类型自身的 clear_kv_cache（candle 上游提供），避免递归到 trait 实现
                    Self::clear_kv_cache(self);
                    Ok(())
                }
            }
        )+
    };
}

/// 不支持清空 KV cache 的模型（Qwen3-MoE、Qwen3-VL）在多轮生成时的统一检查。
///
/// 首次生成（`has_context == false`）时无需清理，返回 Ok；
/// 已经生成过时必须报错——否则会带着上一轮的 KV 继续算，结果错但不报错。
fn reject_multi_turn(has_context: bool, model: &str, reason: &str) -> Result<()> {
    if has_context {
        bail!("{model} 不支持多轮对话：{reason}。\n请每轮重建 ChatPipeline，或改用其它架构。");
    }
    Ok(())
}

pub trait ModelInference {
    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        vision: Option<&VisionInput>,
    ) -> Result<Tensor>;

    /// 清空 KV cache，为下一轮生成做准备。
    ///
    /// 有些模型的缓存藏在私有字段里、上游也没暴露清空接口（Qwen3-MoE、Qwen3-VL），
    /// 这类实现会在「已经生成过」时返回明确的错误，而不是静默地带着旧状态继续算。
    fn reset_kv_cache(&mut self) -> Result<()>;

    /// 是否支持 CUDA Graph 单 token 前向（默认否）。
    ///
    /// 目前只有 Qwen3.5（自实现的混合线性注意力）支持：它的 KV 缓冲容量固定、
    /// 解码步的全部形状与取值都不含 host 常量，可以整步捕获成一张图。
    /// 上游 quantized_qwen3 用的是增长的 ConcatKvCache，不满足这个前提。
    fn supports_graph(&self) -> bool {
        false
    }

    /// CUDA Graph 单 token 前向（只处理 seq_len == 1 的解码步）。
    fn forward_token(&mut self, _token: u32, _index_pos: usize) -> Result<Tensor> {
        anyhow::bail!("该模型不支持 CUDA Graph 单 token 前向")
    }

    /// 如果是视觉模型，返回视觉配置信息
    fn vision_config(&self) -> Option<&VisionConfig> {
        None
    }
}

impl_model_traits!(quantized_qwen3::ModelWeights, qwen3::ModelForCausalLM);

/// GGUFQWenMoE 的封装
pub struct Qwen3MoEModel {
    pub(crate) inner: quantized_qwen3_moe::GGUFQWenMoE,
    /// 上游 GGUFQWenMoE 把每层的 KV cache 藏在私有 `layers` 里，且没有清空接口。
    /// 用它区分「首次生成」和「多轮重生成」：首次无需清理，多轮必须报错，
    /// 否则会带着上一轮的 KV 继续算，结果错但不报错。
    has_context: bool,
}

impl Qwen3MoEModel {
    pub(crate) fn new(inner: quantized_qwen3_moe::GGUFQWenMoE) -> Self {
        Self {
            inner,
            has_context: false,
        }
    }
}

impl ModelInference for Qwen3MoEModel {
    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        _vision: Option<&VisionInput>,
    ) -> anyhow::Result<Tensor> {
        self.has_context = true;
        self.inner.forward(x, index_pos).map_err(anyhow::Error::msg)
    }

    fn reset_kv_cache(&mut self) -> Result<()> {
        reject_multi_turn(
            self.has_context,
            "Qwen3-MoE",
            "candle 0.11 的 GGUFQWenMoE 未暴露清空 KV cache 的接口",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::reject_multi_turn;

    #[test]
    fn first_turn_is_allowed() {
        // 首次生成没有旧状态可清，必须放行，否则模型完全不可用
        assert!(reject_multi_turn(false, "Qwen3-MoE", "测试原因").is_ok());
    }

    #[test]
    fn second_turn_fails_loudly() {
        let err = reject_multi_turn(true, "Qwen3-MoE", "测试原因").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("不支持多轮对话"), "错误信息不够明确: {msg}");
        assert!(msg.contains("Qwen3-MoE"), "错误信息应指明模型: {msg}");
        assert!(
            msg.contains("重建 ChatPipeline"),
            "错误信息应给出解法: {msg}"
        );
    }
}
