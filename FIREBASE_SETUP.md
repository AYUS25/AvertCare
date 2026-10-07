# AvertCare: Firebase Authentication Setup Guide

Since we are building a secure clinical application, hospital staff (clinicians, nurses) must log in before they can request predictions or view patient twins.

Follow these exact steps in the Firebase Console to get the credentials you need to wire up the frontend and backend.

## Part 1: Create the Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/).
2. Click **"Add project"** (or "Create a project").
3. Name the project **"AvertCare"** (or similar) and accept the terms.
4. **Google Analytics**: You can turn this OFF for this hackathon to save time, then click **Create project**.
5. Wait a few seconds for it to provision, then click **Continue** to enter your new project dashboard.

---

## Part 2: Enable Authentication (Email/Password)

1. On the left-hand sidebar menu, click **Build** > **Authentication**.
2. Click the **Get started** button.
3. You will see a list of Sign-in providers. Click on **Email/Password**.
4. Toggle **Enable** for "Email/Password" (leave "Email link" disabled).
5. Click **Save**.
6. Switch to the **Users** tab (next to "Sign-in method" at the top).
7. Click **Add user** and create a dummy clinician account for testing:
   - **Email:** `doctor@avertcare.local`
   - **Password:** `password123`
   *(You will use these credentials to log in on your Next.js dashboard later).*

---

## Part 3: Get Backend Credentials (`FIREBASE_PROJECT_ID`)

The FastAPI backend uses the Firebase Admin SDK to verify the JWT tokens sent by the frontend. It only needs your Project ID.

1. Click the **Gear icon ⚙️** next to "Project Overview" in the top-left corner, and select **Project settings**.
2. Under the **General** tab, look for **Project ID** (it usually looks like `avertcare-1a2b3`).
3. Copy this Project ID.
4. Open your backend environment file: `c:\Drive D\cognizant\AvertCare\backend\.env`.
5. Paste it in:
   ```env
   FIREBASE_PROJECT_ID=avertcare-1a2b3
   ```
*(Note: Because our backend runs in Docker, you'll need to restart the backend container `docker-compose restart backend` for it to pick up this new env variable).*

---

## Part 4: Get Frontend Credentials (Web App Config)

The Next.js frontend needs public API keys to communicate with Firebase to show the login screen and authenticate the user.

1. Still in **Project settings** > **General** tab, scroll down to the **"Your apps"** section.
2. Click the **Web icon (`</>`)** to add a Firebase Web App.
3. App nickname: **"AvertCare Frontend"**.
4. Leave "Firebase Hosting" unchecked, and click **Register app**.
5. Firebase will show you a block of code containing your `firebaseConfig`. It looks like this:
   ```javascript
   const firebaseConfig = {
     apiKey: "AIzaSyB...",
     authDomain: "avertcare-1a2b3.firebaseapp.com",
     projectId: "avertcare-1a2b3",
     storageBucket: "avertcare-1a2b3.firebasestorage.app",
     messagingSenderId: "123456789",
     appId: "1:123456789:web:abcdef123456"
   };
   ```
6. Keep this tab open. You will need to copy these exact values into your Next.js `.env` file (we will create this file in the `frontend/` directory when you're ready).

---

## Part 5: What Happens Next?

Once you have completed the above steps, let me know. 

I will then give you the code to:
1. Initialize the Firebase SDK in your Next.js app.
2. Build a sleek, dark-mode Clinician Login Modal.
3. Intercept all outbound `fetch()` requests from your frontend so they automatically attach the Firebase token (`Authorization: Bearer <token>`) to the FastAPI backend!
