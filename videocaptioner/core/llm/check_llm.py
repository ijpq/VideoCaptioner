"""LLM 连接测试工具"""

import platform
from typing import Literal, Optional

import requests
import requests.exceptions

from videocaptioner.core.llm.client import normalize_base_url


def _extract_api_error_message(response: requests.Response) -> str:
    """从 API 错误响应中提取详细错误信息"""
    try:
        data = response.json()
        # OpenAI-compatible error format: {"error": {"message": "...", "type": "...", "code": "..."}}
        if isinstance(data, dict):
            error = data.get("error", {})
            if isinstance(error, dict):
                parts = []
                if error.get("message"):
                    parts.append(error["message"])
                if error.get("code"):
                    parts.append(f"code={error['code']}")
                if error.get("type"):
                    parts.append(f"type={error['type']}")
                if parts:
                    return " | ".join(parts)
            if "message" in data:
                return str(data["message"])
        return response.text[:500]
    except Exception:
        return response.text[:500]


def _windows_connection_hint(detail: str) -> str:
    """返回 Windows 下常见网络错误的排查提示"""
    hint = "\n\n[Windows 排查建议]"
    if "WinError 10060" in detail:
        hint += "\n• WinError 10060: 连接超时，目标服务器无响应"
        hint += "\n• 请检查 Base URL 是否正确，服务是否正在运行"
        hint += "\n• 检查 Windows 防火墙是否拦截了出站连接"
    elif "WinError 10061" in detail:
        hint += "\n• WinError 10061: 连接被拒绝，目标端口未监听"
        hint += "\n• 请确认本地服务（如 Ollama/LM Studio）已启动"
    elif "WinError 10054" in detail:
        hint += "\n• WinError 10054: 连接被远端强制关闭"
        hint += "\n• 请检查代理软件或 VPN 设置"
    elif "WinError 11001" in detail or "getaddrinfo failed" in detail.lower():
        hint += "\n• 域名解析失败，请检查 Base URL 域名是否拼写正确"
        hint += "\n• 确认网络连接正常，尝试 ping 该域名"
    else:
        hint += "\n• 检查 VPN / 代理设置（系统代理、Clash、V2Ray 等）"
        hint += "\n• 检查 Windows 防火墙出站规则"
        hint += "\n• 检查网络适配器配置"
    return hint


def check_llm_connection(
    base_url: str, api_key: str, model: str
) -> tuple[Literal[True], Optional[str]] | tuple[Literal[False], Optional[str]]:
    """测试 LLM API 连接

    逻辑等价于:
        curl -X POST "<base_url>/chat/completions" \\
             -H "Authorization: Bearer <api_key>" \\
             -H "Content-Type: application/json" \\
             -d '{"model":"<model>","messages":[{"role":"user","content":"Just respond with \\"Hello\\"!"}]}'

    参数:
        base_url: API 基础 URL
        api_key: API 密钥
        model: 模型名称

    返回:
        (是否成功, 错误详情或AI助手的回复)
    """
    base_url = normalize_base_url(base_url)
    api_key = api_key.strip()

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 'Just respond with "Hello"!'},
        ],
    }

    is_windows = platform.system() == "Windows"

    # --- 发送请求（等价于 curl 带 Bearer token）---
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=(15, 30))
    except requests.exceptions.SSLError as e:
        detail = str(e)
        msg = f"SSL 证书错误: {detail}"
        if is_windows:
            msg += (
                "\n\n[Windows SSL 排查建议]"
                "\n• 代理软件（如 Clash）可能拦截了 HTTPS 并替换了证书"
                "\n• 尝试关闭代理或将该地址加入代理白名单"
                "\n• 检查系统证书存储是否损坏（运行 certmgr.msc）"
            )
        msg += f"\n\n请求地址: {url}"
        return False, msg
    except requests.exceptions.ProxyError as e:
        detail = str(e)
        msg = f"代理错误: {detail}"
        if is_windows:
            msg += "\n\n[Windows 排查建议]\n• 请检查系统代理或 VPN 客户端配置"
        msg += f"\n\n请求地址: {url}"
        return False, msg
    except requests.exceptions.ConnectionError as e:
        detail = str(e)
        msg = f"连接失败: {detail}"
        if is_windows:
            msg += _windows_connection_hint(detail)
        else:
            msg += "\n请检查网络连接或 VPN 设置"
        msg += f"\n\n请求地址: {url}"
        return False, msg
    except requests.exceptions.Timeout:
        return False, (
            f"请求超时（连接超时15秒，读取超时30秒）"
            f"\n请求地址: {url}"
            "\n请检查网络连接，或目标服务响应过慢"
        )
    except requests.exceptions.RequestException as e:
        return False, f"请求异常: {type(e).__name__}: {e}\n请求地址: {url}"

    # --- 解析响应 ---
    status = response.status_code

    if status == 200:
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return True, content
        except (KeyError, IndexError, ValueError) as e:
            return False, (
                f"响应解析失败: {e}"
                f"\n原始响应 (前500字符): {response.text[:500]}"
            )

    # HTTP 错误：从响应体提取详细信息
    err_detail = _extract_api_error_message(response)

    if status == 401:
        return False, (
            f"认证失败 (HTTP 401): API Key 无效或已过期"
            f"\n错误详情: {err_detail}"
            f"\n\n请检查 API Key 是否正确复制（不含多余空格）"
        )
    elif status == 403:
        return False, (
            f"权限不足 (HTTP 403): API Key 无权访问该资源"
            f"\n错误详情: {err_detail}"
        )
    elif status == 404:
        return False, (
            f"接口地址不存在 (HTTP 404): 请检查 Base URL 是否正确"
            f"\n请求地址: {url}"
            f"\n错误详情: {err_detail}"
            f"\n\nBase URL 示例: https://api.openai.com/v1"
        )
    elif status == 400:
        return False, (
            f"请求参数错误 (HTTP 400): 请检查模型名称是否正确"
            f"\n错误详情: {err_detail}"
            f"\n模型名称: {model}"
        )
    elif status == 429:
        return False, (
            f"请求频率超限 (HTTP 429): {err_detail}"
            "\n请稍后重试或降低请求频率"
        )
    elif status >= 500:
        return False, (
            f"服务器内部错误 (HTTP {status}): {err_detail}"
            "\n该错误来自 API 服务端，请稍后重试"
        )
    else:
        return False, (
            f"HTTP 错误 {status}: {err_detail}"
            f"\n请求地址: {url}"
        )


def get_available_models(base_url: str, api_key: str) -> list[str]:
    """获取可用的模型列表

    参数:
        base_url: API 基础 URL
        api_key: API 密钥

    返回:
        模型ID列表，按优先级排序
    """
    import openai

    try:
        base_url = normalize_base_url(base_url)
        # 创建OpenAI客户端并获取模型列表
        models = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=5
        ).models.list()

        # 去除非文本模型
        non_text_models = (
            "tts",
            "transcribe",
            "realtime",
            "embedding",
            "vision",
            "audio",
            "search",
            "text-",
            "image",
            "audio",
            "whisper",
            "gpt-3.5",
            "gpt-4-",
        )
        models = [
            model
            for model in models
            if not any(keyword in model.id.lower() for keyword in non_text_models)
        ]

        # 根据不同模型设置权重进行排序
        def get_model_weight(model_name: str) -> int:
            model_name = model_name.lower()
            if model_name.startswith(("gpt-5", "claude-4", "gemini-2", "gemini-3")):
                return 10
            elif model_name.startswith(("gpt-4")):
                return 5
            elif model_name.startswith(("deepseek", "glm", "qwen", "doubao")):
                return 3
            return 0

        sorted_models = sorted(
            [model.id for model in models], key=lambda x: (-get_model_weight(x), x)
        )
        return sorted_models
    except Exception:
        return []
