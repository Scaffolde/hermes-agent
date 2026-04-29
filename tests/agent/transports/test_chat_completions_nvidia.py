from agent.transports.chat_completions import ChatCompletionsTransport


def test_nvidia_deepseek_v4_pro_uses_documented_payload_shape():
    transport = ChatCompletionsTransport()

    kwargs = transport.build_kwargs(
        model="deepseek-ai/deepseek-v4-pro",
        messages=[{"role": "user", "content": "hi"}],
        is_nvidia_nim=True,
        model_lower="deepseek-ai/deepseek-v4-pro",
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )

    assert kwargs["model"] == "deepseek-ai/deepseek-v4-pro"
    assert kwargs["temperature"] == 1
    assert kwargs["top_p"] == 0.95
    assert kwargs["max_tokens"] == 16384
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"thinking": False}


def test_nvidia_deepseek_v4_pro_user_max_tokens_still_keeps_template_kwargs():
    transport = ChatCompletionsTransport()

    kwargs = transport.build_kwargs(
        model="deepseek-ai/deepseek-v4-pro",
        messages=[{"role": "user", "content": "hi"}],
        is_nvidia_nim=True,
        model_lower="deepseek-ai/deepseek-v4-pro",
        max_tokens=256,
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )

    assert kwargs["max_tokens"] == 256
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"thinking": False}


def test_nvidia_other_models_do_not_get_deepseek_v4_pro_template_kwargs():
    transport = ChatCompletionsTransport()

    kwargs = transport.build_kwargs(
        model="deepseek-ai/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        is_nvidia_nim=True,
        model_lower="deepseek-ai/deepseek-v4-flash",
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )

    assert kwargs["max_tokens"] == 16384
    assert "extra_body" not in kwargs or "chat_template_kwargs" not in kwargs.get("extra_body", {})
