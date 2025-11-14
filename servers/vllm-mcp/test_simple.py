#!/usr/bin/env python3
"""
Simple functional tests - basic configuration and initialization
"""

import os


def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")

    try:
        from vllm_mcp.models import MultimodalRequest, TextContent
        from vllm_mcp.providers.dashscope_provider import DashscopeProvider
        from vllm_mcp.server import MultimodalMCPServer
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_server_initialization():
    """Test server initialization"""
    print("\n🔍 Testing server initialization...")

    try:
        from vllm_mcp.server import MultimodalMCPServer

        server = MultimodalMCPServer()
        print(f"✅ Server initialized successfully")
        print(f"   Number of configured providers: {len(server.config.get('providers', []))}")
        print(f"   Initialized providers: {list(server.providers.keys())}")

        if 'dashscope' in server.providers:
            provider = server.providers['dashscope']
            print(f"   Dashscope supported models: {len(provider.supported_models)}")
            print(f"   Default model: qwen-vl-plus")

        return True
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return False


def test_environment_config():
    """Test environment configuration"""
    print("\n🔍 Testing environment configuration...")

    # Check environment variables
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print(f"   DASHSCOPE_API_KEY: {'✅ set' if dashscope_key else '⚠️  not set'}")
    print(f"   OPENAI_API_KEY: {'✅ set' if openai_key else '⚠️  not set'}")

    if dashscope_key:
        print("   ✅ Dashscope configured; ready to test models")
        return True
    else:
        print("   ⚠️  Dashscope API key required to test models")
        return False


def test_configuration_file():
    """Test configuration file"""
    print("\n🔍 Testing configuration file...")

    try:
        if os.path.exists("config.json"):
            import json
            with open("config.json", 'r') as f:
                config = json.load(f)

            print(f"✅ Configuration file loaded successfully")
            print(f"   Transport: {config.get('transport', 'stdio')}")
            print(f"   Host: {config.get('host', 'localhost')}")
            print(f"   Port: {config.get('port', 8080)}")
            print(f"   Provider configs: {len(config.get('providers', []))}")

            return True
        else:
            print("⚠️  config.json does not exist")
            return False
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False


def test_startup_script():
    """Test startup scripts"""
    print("\n🔍 Testing startup scripts...")

    scripts = ["scripts/start.sh", "scripts/start-dev.sh"]

    for script in scripts:
        if os.path.exists(script) and os.access(script, os.X_OK):
            print(f"   ✅ {script} exists and is executable")
        else:
            print(f"   ❌ {script} missing or not executable")
            return False

    return True


def test_examples():
    """Test example files"""
    print("\n🔍 Testing example files...")

    examples = [
        "examples/client_example.py",
        "examples/list_models.py",
        "examples/mcp_client_config.json"
    ]

    for example in examples:
        if os.path.exists(example):
            print(f"   ✅ {example} exists")
        else:
            print(f"   ❌ {example} missing")
            return False

    return True


def main():
    """Run all tests"""
    print("🚀 VLLM MCP Server simple functional tests")
    print("=" * 50)

    tests = [
        ("Basic imports", test_basic_imports),
        ("Server initialization", test_server_initialization),
        ("Environment config", test_environment_config),
        ("Configuration file", test_configuration_file),
        ("Startup scripts", test_startup_script),
        ("Example files", test_examples),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test summary:")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed >= 5:  # at least 5/6 pass
        print("🎉 Basic functional tests passed. Project structure looks good.")
        print("\n🚀 Suggested next steps:")
        print("   1. Ensure .env contains valid API keys")
        print("   2. Start server: ./scripts/start.sh")
        print("   3. List models: uv run python examples/list_models.py")

        if os.getenv("DASHSCOPE_API_KEY"):
            print("   4. Test text generation (Dashscope configured)")

        return True
    else:
        print("⚠️  Some tests failed; please check project configuration.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)