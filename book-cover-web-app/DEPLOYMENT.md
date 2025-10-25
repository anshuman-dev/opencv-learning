# Deployment Guide - Book Cover Analyzer

## Quick Start: Deploy to Render (Free)

### Prerequisites
- GitHub account
- Render account (sign up at render.com - free)

### Step 1: Push to GitHub

```bash
# Navigate to the web app directory
cd /Users/apple/opencv-learning/book-cover-web-app

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Book Cover Analyzer web app"

# Create a new repository on GitHub (e.g., book-cover-analyzer)
# Then connect it:
git remote add origin https://github.com/YOUR_USERNAME/book-cover-analyzer.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** → Select **"Web Service"**
3. **Connect GitHub repository**: `book-cover-analyzer`
4. **Configure:**
   - **Name**: `book-cover-analyzer` (or your choice)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 app:app`
   - **Plan**: Free
5. **Click "Create Web Service"**

**Build time**: ~5-10 minutes (downloads ML models)

**Your app will be live at**: `https://book-cover-analyzer-XXXX.onrender.com`

---

## Alternative: Deploy to Railway

### Step 1: Push to GitHub (same as above)

### Step 2: Deploy on Railway

1. **Go to**: https://railway.app
2. **Click "Start a New Project"**
3. **Select "Deploy from GitHub repo"**
4. **Choose your repository**: `book-cover-analyzer`
5. **Railway auto-detects Python** and deploys automatically

**Cost**: $5/month for Hobby plan (always-on)

---

## Alternative: Deploy to Hugging Face Spaces

### Create a Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 7860

# Run app
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
```

Then:
1. Go to https://huggingface.co/spaces
2. Create new Space → Docker
3. Upload your code
4. Space auto-deploys

---

## Performance Considerations

### Model Loading
- **EasyOCR**: ~400MB download on first run
- **PyTorch ResNet18**: ~45MB download
- **Total startup time**: ~10-20 seconds on free tiers

### Cold Starts
- **Render Free**: Spins down after 15 min inactivity
- **Railway**: Always-on with paid plan
- **Hugging Face**: Stays warm longer

### Memory Usage
- **Minimum RAM**: 1GB
- **Recommended**: 2GB+ for smooth performance
- **Free tiers**: Usually provide 512MB-1GB

---

## Production Optimizations

### 1. Use Gunicorn (Already configured)

```bash
gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 app:app
```

**Why `-w 1` (single worker)?**
- Each worker loads ML models into memory
- ResNet18 + EasyOCR = ~1GB RAM per worker
- Free tiers have limited RAM
- For production with more RAM, use `-w 4`

### 2. Cache Models (Already implemented)

Models are loaded once in `__init__()`, not per request.

### 3. Add Request Limits

Update `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB (already set)
```

### 4. Add Error Handling (Already implemented)

All routes return proper error codes and messages.

---

## Environment Variables

No secrets needed! All models use public pre-trained weights.

Optional:
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
```

---

## Monitoring

### Check if app is running:
```bash
curl https://your-app.onrender.com/health
```

Should return:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "analyzer_ready": true
}
```

---

## Troubleshooting

### Build fails with "Out of memory"
**Solution**: Use Render (better free tier) or upgrade to paid plan

### App times out on first request
**Expected**: Models are loading (~10-20 seconds on cold start)
**Solution**: Keep app warm with scheduled pings, or use paid tier

### "Module not found" errors
**Check**: `requirements.txt` has all dependencies
**Solution**: Verify with `pip freeze > requirements.txt` locally

### Images not uploading
**Check**: `MAX_CONTENT_LENGTH` setting
**Check**: File size < 16MB
**Check**: File type is PNG/JPG/JPEG

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Cold Starts | Best For |
|----------|-----------|-----------|-------------|----------|
| **Render** | 500 hrs/mo | $7/mo | 15 min idle | Hobby projects |
| **Railway** | $5 credit | $5/mo | None (paid) | Always-on apps |
| **Hugging Face** | Unlimited | GPU $0.60/hr | Rare | ML demos |
| **DigitalOcean** | N/A | $6-12/mo | None | Production |

---

## Recommended: Start with Render Free

1. **Free to start**
2. **Automatic HTTPS**
3. **Git-based deployment** (push to GitHub → auto-deploys)
4. **Good for demos and portfolio**

Later, if you need always-on:
- Upgrade to Render paid ($7/mo)
- Or switch to Railway ($5/mo)

---

## Next Steps After Deployment

1. **Test the deployed app** with your book cover images
2. **Share the URL** (e.g., on LinkedIn, portfolio)
3. **Monitor usage** on platform dashboard
4. **Upgrade if needed** based on traffic

---

## Your Live App

Once deployed, you'll have:
- **Public URL**: `https://your-app.onrender.com`
- **Automatic HTTPS**: Secure by default
- **Portfolio-ready**: Show to employers/recruiters

**This is a production-ready ML web application!**
