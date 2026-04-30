# Test Files Directory

Place your test files here for VL model testing.

## Required Files

1. **test_image.jpg** - General image for description testing
   - Any photo, screenshot, or image
   - Recommended size: < 2MB
   - Format: JPEG, PNG, WebP

2. **test_document.jpg** - Image with text for OCR testing
   - Document scan, screenshot with text, or photo of text
   - Should contain readable text in various orientations
   - Test cases: rotated text, low quality, multiple languages
   - Recommended size: < 2MB

3. **test_chart.jpg** - Chart/graph for data extraction
   - Bar chart, line graph, pie chart, or table
   - Should contain numerical data
   - Test if model can extract structured data
   - Recommended size: < 2MB

## Optional Files

4. **test_video.mp4** - Short video clip (if testing video support)
   - Duration: 10-30 seconds
   - Recommended size: < 5MB
   - Format: MP4, AVI, MOV

5. **test_audio.mp3** - Audio file (if testing audio support)
   - Speech recording or conversation
   - Duration: 10-30 seconds
   - Recommended size: < 2MB
   - Format: MP3, WAV

## Where to Get Test Files

### Free Stock Images
- Unsplash: https://unsplash.com
- Pexels: https://pexels.com
- Pixabay: https://pixabay.com

### Document Samples
- Use your own documents (invoices, receipts, forms)
- Screenshot of text from websites
- Photos of printed text

### Chart Samples
- Create simple charts in Excel/Google Sheets
- Screenshot charts from reports
- Use sample charts from data visualization sites

### Video Samples
- Record short clips with your phone
- Use sample videos from Pexels Videos
- Screen recordings

### Audio Samples
- Record voice memos
- Use sample audio from Freesound.org
- Podcast clips

## Privacy Note

⚠️ **Do not use files containing sensitive or personal information!**

Test files will be sent to OpenRouter API for analysis. Use only:
- Public domain content
- Your own non-sensitive content
- Stock images/videos
- Sample data

## File Naming

Keep the exact filenames as listed above, or update the test script accordingly.

## Current Status

- [ ] test_image.jpg
- [ ] test_document.jpg
- [ ] test_chart.jpg
- [ ] test_video.mp4 (optional)
- [ ] test_audio.mp3 (optional)

Once files are added, run:
```bash
python experiments/test_vl_models.py
```
