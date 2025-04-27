import os
import subprocess
import time
import webbrowser
import threading
import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
import uuid
import socket
import sys

# Configuration
PORT = 8000
SCRIPT_PATH = "digital_twin_experiment.py"  # Your original script
ACTIVE_SESSIONS = {}  # Store active sessions and their info

# Check if pyngrok is installed
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    print("pyngrok not found. Installing now...")
    try:
        import pip
        pip.main(['install', 'pyngrok'])
        from pyngrok import ngrok
        NGROK_AVAILABLE = True
        print("pyngrok installed successfully!")
    except Exception as e:
        print(f"Error installing pyngrok: {e}")
        print("Please install it manually with: pip install pyngrok")
        NGROK_AVAILABLE = False

class ErrorHandlingHTTPServer(HTTPServer):
    """HTTP Server that handles Broken Pipe errors gracefully"""
    def handle_error(self, request, client_address):
        # Get the error type
        error_type, error_value, _ = sys.exc_info()
        
        # Handle BrokenPipeError and ConnectionResetError silently
        if issubclass(error_type, (BrokenPipeError, ConnectionResetError)):
            # Just log a simple message without the full traceback
            print(f"Connection from {client_address} closed unexpectedly")
            return
            
        # For all other errors, use the default handling
        super().handle_error(request, client_address)

# Function to find an available port
def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

# Function to create a public URL using ngrok
def create_public_url(port, subdomain=None, domain=None):
    if NGROK_AVAILABLE:
        try:
            # Options for the ngrok tunnel
            options = {}
            
            # Add subdomain if provided (requires paid ngrok account)
            if subdomain:
                options["subdomain"] = subdomain
                
            # Add custom domain if provided (requires paid ngrok account)
            if domain:
                options["domain"] = domain
            
            # Open an ngrok tunnel to the specified port with options
            public_url = ngrok.connect(port, **options).public_url
            print(f"Created public URL: {public_url} -> localhost:{port}")
            return public_url
        except Exception as e:
            print(f"Error creating ngrok tunnel: {e}")
    return None

class RedirectHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to reduce console spam
        if args and isinstance(args[0], str) and args[0].startswith('GET /check'):
            return  # Don't log polling requests
        super().log_message(format, *args)
    
    def send_html_response(self, html):
        """Safely send an HTML response, handling potential connection errors"""
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        except (BrokenPipeError, ConnectionResetError):
            # Client closed connection, log and ignore
            print("Client closed connection while sending response")
        except Exception as e:
            print(f"Error sending response: {e}")
        
    def do_GET(self):
        try:
            if self.path == '/':
                # Show a simple landing page with a button
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Digital Twin Experiment - part of Guneshwar's Thesis</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            max-width: 600px;
                            margin: 0 auto;
                            padding: 20px;
                            text-align: center;
                        }
                        button {
                            background-color: #4CAF50;
                            border: none;
                            color: white;
                            padding: 15px 32px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin: 4px 2px;
                            cursor: pointer;
                            border-radius: 8px;
                        }
                    </style>
                </head>
                <body>
                    <h1>Digital Twin Experiment for Guneshwar's Thesis</h1>
                    <p>Click the button below to start a new session:</p>
                    <button onclick="window.location.href='/launch'">Start New Session</button>
                </body>
                </html>
                """
                self.send_html_response(html)
                
            elif self.path == '/launch':
                # Launch the script directly
                try:
                    # Generate a random port for Gradio
                    gradio_port = find_free_port()
                    session_id = str(uuid.uuid4())[:8]
                    
                    # Launch script with specified port
                    self.launch_script(session_id, gradio_port)
                    
                    # Redirect to the waiting page
                    self.send_response(302)
                    self.send_header('Location', f'/waiting?session={session_id}')
                    self.end_headers()
                    
                except Exception as e:
                    try:
                        self.send_response(500)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(f"Error launching script: {str(e)}".encode())
                    except (BrokenPipeError, ConnectionResetError):
                        print("Connection closed while sending error response")
                    
            elif self.path.startswith('/waiting'):
                # Show a waiting page that checks for the Gradio URL
                session_id = self.path.split('=')[1] if '=' in self.path else None
                
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Launching Session...</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            max-width: 600px;
                            margin: 0 auto;
                            padding: 20px;
                            text-align: center;
                        }}
                        .loader {{
                            border: 16px solid #f3f3f3;
                            border-top: 16px solid #3498db;
                            border-radius: 50%;
                            width: 120px;
                            height: 120px;
                            animation: spin 2s linear infinite;
                            margin: 20px auto;
                        }}
                        @keyframes spin {{
                            0% {{ transform: rotate(0deg); }}
                            100% {{ transform: rotate(360deg); }}
                        }}
                    </style>
                    <script>
                        let attempts = 0;
                        const maxAttempts = 30; // 30 seconds timeout
                        
                        function checkUrl() {{
                            fetch('/check?session={session_id}')
                                .then(response => response.json())
                                .then(data => {{
                                    if (data.url) {{
                                        window.location.href = data.url;
                                    }} else {{
                                        attempts++;
                                        if (attempts < maxAttempts) {{
                                            setTimeout(checkUrl, 1000);
                                        }} else {{
                                            document.getElementById('loader').style.display = 'none';
                                            document.getElementById('timeout').style.display = 'block';
                                        }}
                                    }}
                                }})
                                .catch(error => {{
                                    attempts++;
                                    if (attempts < maxAttempts) {{
                                        setTimeout(checkUrl, 1000);
                                    }} else {{
                                        document.getElementById('loader').style.display = 'none';
                                        document.getElementById('timeout').style.display = 'block';
                                    }}
                                }});
                        }}
                        
                        window.onload = function() {{
                            checkUrl();
                        }};
                    </script>
                </head>
                <body>
                    <h1>Launching Your Session</h1>
                    <p>Please wait while we set up your Digital Twin Experiment...</p>
                    <h3>Just interact with the chatbot as you normally would with any other interaction</h3>
                    <p>Use <h4>Clicked on the Link button</h4> when and if you click on the link in the chat</p>
                    <p>Use <h4>End Session button </h4>whenenever you want to end the session</p>
                    <p><h4>Submit Feedback</h4> when you're done with the session</p>
                    <div id="loader" class="loader"></div>
                    <div id="timeout" style="display:none;">
                        <p style="color: red;">It's taking longer than expected. The session might still be starting up.</p>
                        <p>You can try <a href="/waiting?session={session_id}">refreshing this page</a> or going <a href="/">back to the main page</a>.</p>
                    </div>
                </body>
                </html>
                """
                self.send_html_response(html)
                
            elif self.path.startswith('/check'):
                # Check if the Gradio URL is available and return it
                session_id = self.path.split('=')[1] if '=' in self.path else None
                
                try:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    response = {"url": None}
                    if session_id in ACTIVE_SESSIONS and ACTIVE_SESSIONS[session_id]["url"]:
                        response["url"] = ACTIVE_SESSIONS[session_id]["url"]
                        
                    self.wfile.write(json.dumps(response).encode())
                except (BrokenPipeError, ConnectionResetError):
                    # Client closed connection, ignore
                    pass
            
            else:
                try:
                    self.send_response(404)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'Not Found')
                except (BrokenPipeError, ConnectionResetError):
                    # Client closed connection, ignore
                    pass
                    
        except (BrokenPipeError, ConnectionResetError):
            # Client closed connection, ignore
            print("Client closed connection during request handling")
        except Exception as e:
            print(f"Error handling request: {e}")

    def launch_script(self, session_id, gradio_port):
        """Launch the script directly with a specified port"""
        
        print(f"Launching script for session {session_id} on port {gradio_port}")
        
        # Create a modified version of the script with the port specified
        temp_script_path = f"temp_script_{session_id}.py"
        
        with open(SCRIPT_PATH, "r") as src_file:
            script_content = src_file.read()
            
            # First, check for existing launch parameters
            if "demo.launch(" in script_content:
                # If there's an existing call, we need to carefully modify it
                launch_pattern = r"demo\.launch\s*\(([^)]*)\)"
                launch_matches = re.findall(launch_pattern, script_content)
                
                if launch_matches:
                    for match in launch_matches:
                        original_params = match.strip()
                        
                        # Remove share=True if it exists
                        new_params = re.sub(r'share\s*=\s*True', '', original_params)
                        new_params = re.sub(r'share\s*=\s*False', '', new_params)
                        new_params = re.sub(r',,', ',', new_params)  # Clean up double commas
                        new_params = re.sub(r'^,|,$', '', new_params)  # Clean up leading/trailing commas
                        
                        # Check if already has server_port
                        if "server_port" in new_params:
                            new_params = re.sub(r'server_port\s*=\s*[^,)]+', f'server_port={gradio_port}', new_params)
                        else:
                            if new_params:
                                new_params = f'server_port={gradio_port}, {new_params}'
                            else:
                                new_params = f'server_port={gradio_port}'
                                
                        # Ensure we have share=False to prevent Gradio from creating its own public URL
                        if "share=" not in new_params:
                            new_params = f'{new_params}, share=False'
                                
                        script_content = script_content.replace(
                            f"demo.launch({original_params})", 
                            f"demo.launch({new_params})"
                        )
            else:
                # If no launch call is found, add one at the end
                script_content += f"\n\nif __name__ == '__main__':\n    demo.launch(server_port={gradio_port}, share=False)\n"
            
            with open(temp_script_path, "w") as dest_file:
                dest_file.write(script_content)
        
        # Start the process directly (no terminal)
        process = subprocess.Popen(
            ["python", temp_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False
        )
        
        # Store process info
        ACTIVE_SESSIONS[session_id] = {
            "process": process,
            "url": None,
            "started_at": time.time(),
            "temp_script": temp_script_path,
            "port": gradio_port
        }
        
        # Create a public URL for this Gradio instance
        def create_tunnel():
            # Wait a bit for Gradio to start
            time.sleep(5)
            
            try:
                if NGROK_AVAILABLE:
                    # Create our own ngrok tunnel to the Gradio port
                    public_url = create_public_url(gradio_port)
                    if public_url:
                        ACTIVE_SESSIONS[session_id]["url"] = public_url
                        print(f"Session {session_id} accessible at: {public_url}")
                else:
                    # Fallback to local URL
                    local_url = f"http://localhost:{gradio_port}"
                    ACTIVE_SESSIONS[session_id]["url"] = local_url
                    print(f"Session {session_id} accessible at: {local_url}")
            except Exception as e:
                print(f"Error setting up tunnel for session {session_id}: {e}")
        
        # Start tunnel creation in a separate thread
        tunnel_thread = threading.Thread(target=create_tunnel)
        tunnel_thread.daemon = True
        tunnel_thread.start()
        
        # Function to run in a separate thread to monitor output (for logging)
        def monitor_process(process, session_id):
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                try:
                    line = line.decode('utf-8', errors='ignore').strip()
                    if line:  # Only print non-empty lines
                        print(f"[{session_id}] {line}")
                except:
                    pass
                
            # Clean up the temporary script when done
            try:
                os.remove(ACTIVE_SESSIONS[session_id]["temp_script"])
                print(f"Cleaned up temporary script for session {session_id}")
            except:
                pass
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_process, args=(process, session_id))
        monitor_thread.daemon = True
        monitor_thread.start()

def setup_ngrok():
    """Setup ngrok for public URL access"""
    if not NGROK_AVAILABLE:
        return None
    
    try:
        # Check if auth token is set
        try:
            # This will throw an exception if auth token isn't set
            ngrok.get_ngrok_process().api_url
        except Exception:
            print("\n=== NGROK SETUP REQUIRED ===")
            print("To create public URLs, you need to set up ngrok.")
            print("1. Sign up for a free account at https://ngrok.com")
            print("2. Get your authtoken from the ngrok dashboard")
            
            auth_token = input("Enter your ngrok authtoken (or press Enter to use local URLs only): ").strip()
            if auth_token:
                ngrok.set_auth_token(auth_token)
                print("Authtoken set successfully!")
            else:
                print("No authtoken provided. Using local URLs only.")
                return None
        
        # Create a public URL for the launcher
        public_url = create_public_url(PORT)
        if public_url:
            print(f"\n=== SHARE THIS LINK ===")
            print(f"Public URL: {public_url}")
            print(f"Anyone with this link can access your Digital Twin Experiment")
            print("======================\n")
            return public_url
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
    
    return None

def main():
    # Check if original script exists
    if not os.path.exists(SCRIPT_PATH):
        print(f"Error: Could not find the script: {SCRIPT_PATH}")
        print(f"Please make sure to save your script as {SCRIPT_PATH} in the same directory.")
        return
    
    # Setup ngrok for public URL
    public_url = setup_ngrok()
    
    # Start server with enhanced error handling
    server = ErrorHandlingHTTPServer(("0.0.0.0", PORT), RedirectHandler)
    print(f"Server started on http://localhost:{PORT}")
    if public_url:
        print(f"Public URL: {public_url}")
    else:
        print("No public URL available. Using local URLs only.")
    
    try:
        # Open the browser automatically
        webbrowser.open(f"http://localhost:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        # Clean up all processes when shutting down
        for session_id, session in ACTIVE_SESSIONS.items():
            try:
                if session["process"].poll() is None:
                    print(f"Terminating session {session_id}")
                    session["process"].terminate()
                
                # Clean up temp script if it exists
                if "temp_script" in session and os.path.exists(session["temp_script"]):
                    os.remove(session["temp_script"])
            except Exception as e:
                print(f"Error cleaning up: {e}")
        
        # Clean up ngrok tunnels
        if NGROK_AVAILABLE:
            try:
                ngrok.kill()
            except:
                pass

if __name__ == "__main__":
    main()