import json
import subprocess
import os
import sys
import threading
import queue
import time

# Configuration
MCP_SERVERS_FILE = "mcp_servers.json"
TIMEOUT_SECONDS = 15

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = ".env"
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Handle simple quoting
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    # Only set if not already set (optional, but usually env vars > .env)
                    # Or force overwrite? User wants to use .env, usually that means overlay.
                    # Let's force set to ensure .env values are used.
                    os.environ[key] = value

def load_config():
    if not os.path.exists(MCP_SERVERS_FILE):
        print(f"Error: {MCP_SERVERS_FILE} not found.")
        return None
    with open(MCP_SERVERS_FILE, 'r') as f:
        return json.load(f)

def test_server(server_name, config):
    if config.get("disabled", False):
        return "SKIPPED", "Disabled in config"

    command = config.get("command")
    args = config.get("args", [])
    env_vars = config.get("env", {})
    
    # Merge current environment with server specific env
    full_env = os.environ.copy()
    full_env.update(env_vars)
    
    # Ensure we are using the absolute path for the command if it looks like a file path
    if "/" in command and not os.path.isabs(command):
        command = os.path.abspath(command)
    

    # Construct the full command
    full_cmd = [command] + args
    
    print(f"Testing {server_name}...", end=" ", flush=True)
    
    process = None
    try:
        process = subprocess.Popen(
            full_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            bufsize=1 # Line buffered
        )
        
        # Prepare initialize message
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-tester", "version": "1.0"}
            }
        }
        
        input_str = json.dumps(init_msg) + "\n"
        
        def read_output(p, q):
            try:
                # Read just the first line
                line = p.stdout.readline()
                if line:
                    q.put(line)
            except Exception:
                pass

        output_queue = queue.Queue()
        t = threading.Thread(target=read_output, args=(process, output_queue))
        t.daemon = True
        t.start()
        
        # Send input
        try:
            process.stdin.write(input_str)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
             # Check stderr if possible
            _, stderr_out = process.communicate(timeout=1)
            return "FAILED", f"Process exited early: {stderr_out.strip() if stderr_out else str(e)}"

        # Wait for response
        try:
            line = output_queue.get(timeout=TIMEOUT_SECONDS)
            # Parse JSON
            try:
                response = json.loads(line)
                if "result" in response:
                    return "SUCCESS", "Initialization successful"
                elif "error" in response:
                     return "FAILED", f"Server error: {response['error'].get('message', 'Unknown error')}"
                else:
                    return "FAILED", "Invalid JSON-RPC response structure"
            except json.JSONDecodeError:
                return "FAILED", f"Invalid JSON received: {line.strip()[:500]}..."
                
        except queue.Empty:
             # Check if process is still running
            if process.poll() is not None:
                _, stderr_out = process.communicate()
                return "FAILED", f"Process exited with code {process.returncode}. Stderr: {stderr_out.strip()[:2000]}..."
            return "FAILED", "Timeout waiting for response"

    except FileNotFoundError:
        return "FAILED", f"Command not found: {command}"
    except Exception as e:
        return "FAILED", str(e)
    finally:
        if process:
            try:
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass

def main():
    load_env_file()
    config = load_config()
    if not config:
        return

    servers = config.get("servers", {})
    results = []

    print(f"Found {len(servers)} servers. Starting tests...\n")

    for name, server_conf in servers.items():
        status, message = test_server(name, server_conf)
        results.append((name, status, message))
        # Print immediate result
        print(f"[{status}]")
        if status == "FAILED":
            print(f"  Reason: {message}")

    print("\n" + "="*140)
    print(f"{'Server':<25} | {'Status':<10} | {'Message'}")
    print("-" * 140)
    for name, status, message in results:
        # Truncate message if too long
        display_msg = (message[:100] + '..') if len(message) > 100 else message
        print(f"{name:<25} | {status:<10} | {display_msg}")
    print("="*140)

if __name__ == "__main__":
    main()

