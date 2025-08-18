import React, { useState } from "react";

// Auto-detect backend (local vs deployed)
const BACKEND_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://your-deployed-backend-url.com"; // Change if you deploy later

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [pdfFile, setPdfFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");

  // Handle PDF upload
  const handlePdfUpload = async () => {
    if (!pdfFile) {
      setUploadMessage("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", pdfFile);

    try {
      const response = await fetch(`${BACKEND_URL}/upload-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      setUploadMessage(data.message);
    } catch (error) {
      setUploadMessage("Upload failed: Could not reach backend.");
    }
  };

  // Handle asking a question
  const askQuestion = async () => {
    if (!question.trim()) {
      setAnswer("Please enter a question.");
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error("Query failed");
      }

      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      setAnswer("Error: Could not fetch answer.");
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>📄 RAG PDF Search</h2>

      {/* PDF Upload Section */}
      <div style={{ marginBottom: "20px" }}>
        <h3>Upload PDF</h3>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setPdfFile(e.target.files[0])}
        />
        <button onClick={handlePdfUpload} style={{ marginLeft: "10px" }}>
          Upload
        </button>
        <p>{uploadMessage}</p>
      </div>

      {/* Question Answer Section */}
      <div>
        <h3>Ask a Question</h3>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{ width: "300px", marginRight: "10px" }}
        />
        <button onClick={askQuestion}>Ask</button>
        <p>
          <b>Answer:</b> {answer}
        </p>
      </div>
    </div>
  );
}

export default App;
