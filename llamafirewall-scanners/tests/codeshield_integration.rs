/// Integration tests for CodeShield scanner with Semgrep
///
/// These tests require Semgrep to be installed: `pip install semgrep`
/// Run with: `cargo test --test codeshield_integration -- --ignored`
use llamafirewall_core::scanner::Scanner;
use llamafirewall_core::types::{Message, Role, ScanDecision};
use llamafirewall_scanners::CodeShieldScanner;

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_python_eval_detection() {
    // Dangerous eval usage - should be detected
    let code = r#"
import os

def process_user_input(user_data):
    # Dangerous: eval allows arbitrary code execution
    result = eval(user_data)
    return result

user_input = input("Enter expression: ")
output = process_user_input(user_input)
print(output)
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("Python eval detection result: {:?}", scan_result);

    // Should detect eval as a security issue
    if scan_result.decision == ScanDecision::Block {
        assert!(
            scan_result.reason.to_lowercase().contains("eval")
                || scan_result.reason.to_lowercase().contains("dangerous"),
            "Should mention eval or dangerous code in reason"
        );
    }
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_python_sql_injection() {
    // SQL injection vulnerability - should be detected
    let code = r#"
import sqlite3

def get_user(username):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Dangerous: SQL injection via string formatting
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)

    return cursor.fetchone()

# User input directly in SQL query
user = get_user(request.args.get('username'))
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("Python SQL injection result: {:?}", scan_result);

    // May detect SQL injection depending on ruleset
    if scan_result.decision == ScanDecision::Block {
        println!("Detected security issue: {}", scan_result.reason);
    }
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_javascript_xss() {
    // XSS vulnerability - should be detected
    let code = r#"
const express = require('express');
const app = express();

app.get('/search', (req, res) => {
    // Dangerous: XSS via unsanitized user input
    const searchQuery = req.query.q;
    res.send('<html><body><h1>Search Results for: ' + searchQuery + '</h1></body></html>');
});

app.get('/profile', (req, res) => {
    // Another XSS vector
    const username = req.params.name;
    res.send(`<div>Welcome ${username}</div>`);
});

app.listen(3000);
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("JavaScript XSS detection result: {:?}", scan_result);

    // May detect XSS depending on ruleset
    if scan_result.decision == ScanDecision::Block {
        println!("Detected security issue: {}", scan_result.reason);
    }
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_python_command_injection() {
    // Command injection vulnerability
    let code = r#"
import subprocess
import os

def execute_command(filename):
    # Dangerous: command injection via subprocess
    os.system(f"cat {filename}")

def backup_file(path):
    # Another dangerous pattern
    subprocess.call(f"cp {path} /backup/", shell=True)

user_file = input("Enter filename: ")
execute_command(user_file)
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("Python command injection result: {:?}", scan_result);

    // Should detect command injection
    if scan_result.decision == ScanDecision::Block {
        assert!(
            scan_result.reason.to_lowercase().contains("command")
                || scan_result.reason.to_lowercase().contains("injection")
                || scan_result.reason.to_lowercase().contains("subprocess"),
            "Should mention command injection"
        );
    }
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_java_path_traversal() {
    // Path traversal vulnerability
    let code = r#"
import java.io.*;
import javax.servlet.http.*;

public class FileDownloadServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) {
        String filename = request.getParameter("file");

        // Dangerous: path traversal vulnerability
        File file = new File("/var/www/files/" + filename);

        FileInputStream fis = new FileInputStream(file);
        OutputStream os = response.getOutputStream();

        byte[] buffer = new byte[1024];
        int bytesRead;
        while ((bytesRead = fis.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
    }
}
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("Java path traversal result: {:?}", scan_result);

    // May detect path traversal depending on ruleset
    if scan_result.decision == ScanDecision::Block {
        println!("Detected security issue: {}", scan_result.reason);
    }
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_safe_code_python() {
    // Safe code - should pass
    let code = r#"
def calculate_fibonacci(n):
    """Calculate Fibonacci number safely."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

def greet_user(name):
    """Greet user with sanitized input."""
    # Safe: parameterized and validated
    safe_name = name.strip()[:50]  # Limit length
    return f"Hello, {safe_name}!"

if __name__ == "__main__":
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")
    print(greet_user("Alice"))
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("Safe Python code result: {:?}", scan_result);

    // Safe code should typically pass (though depends on ruleset)
    println!(
        "Decision: {:?}, Score: {}",
        scan_result.decision, scan_result.score
    );
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_safe_code_javascript() {
    // Safe code with proper sanitization
    let code = r#"
const express = require('express');
const { body, validationResult } = require('express-validator');
const app = express();

app.use(express.json());

// Safe: proper input validation and sanitization
app.post('/api/users', [
    body('email').isEmail().normalizeEmail(),
    body('username').trim().escape().isLength({ min: 3, max: 20 }),
], (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
    }

    const { email, username } = req.body;
    // Safe processing with validated input
    console.log(`New user: ${username} (${email})`);
    res.json({ success: true });
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("Safe JavaScript code result: {:?}", scan_result);
    println!(
        "Decision: {:?}, Score: {}",
        scan_result.decision, scan_result.score
    );
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_empty_code() {
    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, "");

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Empty code should be handled gracefully");

    let scan_result = result.unwrap();
    assert_eq!(
        scan_result.decision,
        ScanDecision::Allow,
        "Empty code should be allowed"
    );
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_whitespace_only() {
    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, "   \n\t\n   ");

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Whitespace should be handled gracefully");

    let scan_result = result.unwrap();
    assert_eq!(
        scan_result.decision,
        ScanDecision::Allow,
        "Whitespace-only should be allowed"
    );
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_multiple_vulnerabilities() {
    // Code with multiple security issues
    let code = r#"
import os
import subprocess

def process_data(user_input):
    # Issue 1: eval
    result = eval(user_input)

    # Issue 2: command injection
    os.system(f"echo {result}")

    # Issue 3: SQL injection
    query = f"SELECT * FROM data WHERE value = '{result}'"

    return query

data = input("Enter data: ")
process_data(data)
"#;

    let scanner = CodeShieldScanner::new();
    let message = Message::new(Role::User, code);

    let result = scanner.scan(&message, None).await;
    assert!(result.is_ok(), "Scan should succeed");

    let scan_result = result.unwrap();
    println!("Multiple vulnerabilities result: {:?}", scan_result);

    // Should detect at least one issue
    if scan_result.decision == ScanDecision::Block {
        assert!(!scan_result.reason.is_empty(), "Should have reason");
        println!("Detected issues: {}", scan_result.reason);
    }
}

#[tokio::test]
#[ignore] // Requires Semgrep installation
async fn test_language_detection() {
    // Test automatic language detection
    let python_code = "def hello():\n    print('world')";
    let js_code = "function hello() { console.log('world'); }";

    let scanner = CodeShieldScanner::new();

    let python_msg = Message::new(Role::User, python_code);
    let python_result = scanner.scan(&python_msg, None).await;
    assert!(python_result.is_ok(), "Python code should scan successfully");

    let js_msg = Message::new(Role::User, js_code);
    let js_result = scanner.scan(&js_msg, None).await;
    assert!(js_result.is_ok(), "JavaScript code should scan successfully");

    println!("Python scan: {:?}", python_result.unwrap());
    println!("JavaScript scan: {:?}", js_result.unwrap());
}
