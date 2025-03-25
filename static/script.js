document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');

    if (fileInput.files.length === 0) {
        resultDiv.innerHTML = '<p style="color: red;">Please select a file.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        resultDiv.innerHTML = '<p>Processing... Please wait.</p>';
        const response = await fetch('/uploads', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'Forensic_Analysis_Report.pdf';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            resultDiv.innerHTML = '<p style="color: green;">Report generated successfully. Download started.</p>';
        } else {
            resultDiv.innerHTML = '<p style="color: red;">Error generating report. Please try again.</p>';
        }
    } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
});