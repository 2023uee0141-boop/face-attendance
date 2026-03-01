/**
 * Python Script Runner Utility
 * 
 * Executes Python scripts from Node.js using child_process.
 * Handles JSON communication between Node and Python.
 * 
 * All Python scripts receive input as command-line arguments
 * and return JSON output via stdout.
 */

const { spawn } = require('child_process');
const path = require('path');

// Path to the Python scripts directory
const PYTHON_DIR = path.join(__dirname, '..', 'python');

// Python executable - use venv if available
const getPythonPath = () => {
  const venvPython = path.join(PYTHON_DIR, '..', 'venv', 'bin', 'python');
  return process.env.PYTHON_PATH || venvPython;
};

/**
 * Execute a Python script and return parsed JSON output
 * 
 * @param {string} scriptName - Name of the Python script (e.g., 'detect.py')
 * @param {string[]} args - Command-line arguments to pass
 * @returns {Promise<object>} - Parsed JSON output from the script
 */
const runPythonScript = (scriptName, args = []) => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(PYTHON_DIR, scriptName);
    const pythonPath = getPythonPath();

    console.log(`[PYTHON] Running: ${pythonPath} ${scriptPath} ${args.join(' ')}`);

    const process = spawn(pythonPath, [scriptPath, ...args], {
      cwd: PYTHON_DIR,
      env: { ...global.process.env, PYTHONUNBUFFERED: '1' },
    });

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    process.stderr.on('data', (data) => {
      stderr += data.toString();
      // Log Python stderr for debugging (includes model loading info)
      console.log(`[PYTHON STDERR] ${data.toString().trim()}`);
    });

    process.on('close', (code) => {
      if (code !== 0) {
        console.error(`[PYTHON] Script ${scriptName} exited with code ${code}`);
        console.error(`[PYTHON] stderr: ${stderr}`);
        return reject(new Error(`Python script '${scriptName}' failed: ${stderr || 'Unknown error'}`));
      }

      try {
        // Extract JSON from stdout - try last line first (most reliable),
        // then fall back to regex matching anywhere in output
        const lines = stdout.trim().split('\n').filter(l => l.trim());
        let result = null;

        // Try parsing the last line as JSON (our scripts print JSON as last line)
        for (let i = lines.length - 1; i >= 0; i--) {
          try {
            result = JSON.parse(lines[i].trim());
            break;
          } catch (e) {
            // Not valid JSON, try previous line
          }
        }

        if (!result) {
          return reject(new Error(`No JSON output from ${scriptName}. Output: ${stdout}`));
        }

        console.log(`[PYTHON] ${scriptName} completed successfully`);
        resolve(result);
      } catch (parseError) {
        console.error(`[PYTHON] Failed to parse output from ${scriptName}:`, stdout);
        reject(new Error(`Failed to parse Python output: ${parseError.message}`));
      }
    });

    process.on('error', (err) => {
      console.error(`[PYTHON] Failed to start ${scriptName}:`, err.message);
      reject(new Error(`Failed to start Python script: ${err.message}`));
    });
  });
};

module.exports = { runPythonScript };
