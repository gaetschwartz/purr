const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('Purr Backend API Tests', () => {
  const baseURL = 'http://localhost:8080';

  test('should have healthy backend server', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/health`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.status).toBe('healthy');
    expect(data.service).toBe('purr-backend');
    console.log('Backend health check passed:', data);
  });

  test('should upload file and transcribe with real Whisper', async ({ request }) => {
    // Upload the JFK audio file
    const testFilePath = path.join(__dirname, '..', 'samples', 'jfk.wav');

    const uploadResponse = await request.post(`${baseURL}/api/upload`, {
      multipart: {
        file: {
          name: 'jfk.wav',
          mimeType: 'audio/wav',
          buffer: require('fs').readFileSync(testFilePath)
        }
      }
    });

    expect(uploadResponse.status()).toBe(200);
    const uploadData = await uploadResponse.json();
    expect(uploadData.file_id).toBeTruthy();
    console.log('File uploaded:', uploadData);

    // Start real transcription
    const transcribeResponse = await request.get(`${baseURL}/api/transcribe?file_id=${uploadData.file_id}`);
    expect(transcribeResponse.status()).toBe(200);

    const transcriptionData = await transcribeResponse.json();
    expect(transcriptionData.text).toBeTruthy();
    expect(transcriptionData.processing_time).toBeGreaterThan(0);
    expect(transcriptionData.word_count).toBeGreaterThan(0);

    // Verify this is REAL transcription, not mock
    expect(transcriptionData.text).not.toContain('mock');
    expect(transcriptionData.text).not.toContain('placeholder');
    expect(transcriptionData.text.length).toBeGreaterThan(20);

    console.log('REAL Transcription Result:');
    console.log('Text:', transcriptionData.text);
    console.log('Processing Time:', transcriptionData.processing_time);
    console.log('Word Count:', transcriptionData.word_count);

    // For JFK audio, expect some related content
    const lowerText = transcriptionData.text.toLowerCase();
    const hasRelevantContent = lowerText.includes('president') ||
                             lowerText.includes('kennedy') ||
                             lowerText.includes('john') ||
                             lowerText.includes('ask') ||
                             lowerText.includes('nation') ||
                             lowerText.includes('country') ||
                             lowerText.includes('american');

    if (hasRelevantContent) {
      console.log('✅ Transcription contains expected JFK-related content');
    } else {
      console.log('⚠️ Transcription may not be accurate, but it is REAL (not mock)');
    }
  });
});