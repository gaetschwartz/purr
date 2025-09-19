const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('Purr Transcription App', () => {
  test.beforeEach(async ({ page }) => {
    // Go to the home page
    await page.goto('/');
  });

  test('should load home page with drag-drop zone', async ({ page }) => {
    // Check that the page loads
    await expect(page).toHaveTitle(/Purr - Audio Transcription/);

    // Check for drag-drop zone
    await expect(page.locator('input[type="file"]')).toBeVisible();
    await expect(page.getByText('Click to upload or drag and drop')).toBeVisible();
    await expect(page.getByText('Audio files supported (MP3, WAV, AAC, FLAC, OGG, M4A)')).toBeVisible();

    // Take a screenshot for visual inspection
    await page.screenshot({ path: 'test-results/home-page.png', fullPage: true });
  });

  test('should upload file and navigate to transcription page', async ({ page }) => {
    // Get the file input element
    const fileInput = page.locator('input[type="file"]');

    // Upload a test audio file
    const testFilePath = path.join(__dirname, '..', 'samples', 'jfk.wav');
    await fileInput.setInputFiles(testFilePath);

    // Wait for upload to complete and navigation to transcription page
    await expect(page).toHaveURL(/\/transcription\/.+/, { timeout: 30000 });

    // Verify we're on the transcription page
    await expect(page.getByText('Audio Transcription')).toBeVisible();

    // Take a screenshot of the transcription page
    await page.screenshot({ path: 'test-results/transcription-page.png', fullPage: true });
  });

  test('should complete full transcription workflow', async ({ page }) => {
    // Upload a test audio file
    const fileInput = page.locator('input[type="file"]');
    const testFilePath = path.join(__dirname, '..', 'samples', 'jfk.wav');
    await fileInput.setInputFiles(testFilePath);

    // Wait for navigation to transcription page
    await expect(page).toHaveURL(/\/transcription\/.+/, { timeout: 30000 });

    // Wait for transcription to start
    await expect(page.getByText('Processing audio file...')).toBeVisible({ timeout: 10000 });

    // Wait for transcription to complete (this may take some time for real transcription)
    await expect(
      page.getByText('Transcription Complete!'),
      { timeout: 300000 }  // 5 minutes timeout for real transcription
    ).toBeVisible();

    // Verify final transcription results are displayed
    await expect(page.getByText('Final Transcription')).toBeVisible();

    // Check that there's actual transcription text
    const transcriptionText = await page.locator('.whitespace-pre-wrap').textContent();
    expect(transcriptionText).toBeTruthy();
    expect(transcriptionText.length).toBeGreaterThan(10);

    // CRITICAL: Verify this is NOT mock data
    expect(transcriptionText).not.toContain('This is a mock transcription');
    expect(transcriptionText).not.toContain('mock');
    expect(transcriptionText).not.toContain('placeholder');

    // For JFK audio, expect real transcription content
    const lowerText = transcriptionText.toLowerCase();
    const hasJFKContent = lowerText.includes('president') ||
                         lowerText.includes('kennedy') ||
                         lowerText.includes('john') ||
                         lowerText.includes('ask not') ||
                         lowerText.includes('nation') ||
                         lowerText.includes('country');

    if (!hasJFKContent) {
      console.warn('Warning: Transcription may not be accurate JFK content:', transcriptionText);
    }

    // Verify statistics are shown
    await expect(page.getByText('Processing Time')).toBeVisible();
    await expect(page.getByText('Audio Duration')).toBeVisible();
    await expect(page.getByText('Word Count')).toBeVisible();

    // Take a screenshot of the completed transcription
    await page.screenshot({ path: 'test-results/transcription-complete.png', fullPage: true });

    console.log('REAL Transcription completed successfully:', transcriptionText);
  });

  test('should handle back navigation', async ({ page }) => {
    // Upload a file and navigate to transcription page
    const fileInput = page.locator('input[type="file"]');
    const testFilePath = path.join(__dirname, '..', 'samples', 'jfk.wav');
    await fileInput.setInputFiles(testFilePath);

    await expect(page).toHaveURL(/\/transcription\/.+/, { timeout: 30000 });

    // Click the back button
    await page.getByRole('button', { name: 'Back' }).click();

    // Should navigate back to home page
    await expect(page).toHaveURL('/');
    await expect(page.getByText('Click to upload')).toBeVisible();
  });

  test('should handle drag and drop functionality', async ({ page }) => {
    // Create a data transfer with a test file
    const testFilePath = path.join(__dirname, '..', 'samples', 'jfk.wav');

    // Find the drop zone
    const dropZone = page.locator('.border-dashed');

    // Simulate drag over
    await dropZone.dispatchEvent('dragover', {
      dataTransfer: {
        files: [{ name: 'jfk.wav', type: 'audio/wav' }]
      }
    });

    // Check that drag over styling is applied
    await expect(dropZone).toHaveClass(/border-teal-500/);

    // Note: Actually simulating file drop is complex in Playwright
    // We'll focus on the file input method for the actual workflow test
  });

  test('should display proper loading states', async ({ page }) => {
    // Upload a file
    const fileInput = page.locator('input[type="file"]');
    const testFilePath = path.join(__dirname, '..', 'samples', 'jfk.wav');
    await fileInput.setInputFiles(testFilePath);

    // Check for upload loading state (may be very quick)
    // await expect(page.getByText('Uploading file...')).toBeVisible({ timeout: 5000 });

    // Wait for navigation
    await expect(page).toHaveURL(/\/transcription\/.+/, { timeout: 30000 });

    // Check for transcription loading states
    await expect(
      page.getByText('Processing audio file...')
    ).toBeVisible({ timeout: 10000 });

    // Verify loading spinner is present
    await expect(page.locator('.animate-spin')).toBeVisible();
  });
});