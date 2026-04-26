# Retail-IQ Dashboard Architecture & Dissident's Note

## The Architecture Map

```text
dashboard/
├── app.py           (Flask API + Error Handling + JWT Middleware)
├── db.py            (MongoDB Pymongo Wrapper + Indexing)
├── models.py        (Pydantic Request/Response Data Contracts)
├── seed_db.py       (Pipeline Mock Data Generator)
├── static/          (Empty - Handled directly via CDNs)
└── templates/
    └── index.html   (Single Page App: Alpine.js + Chart.js + Vanilla CSS)
```

## The Dissident's Note

**"We intentionally stripped out React, Webpack, ORMs (SQLAlchemy/MongoEngine), and application factories, keeping the entire frontend state localized in Alpine.js and database queries as raw dictionaries, because adding 50MB of `node_modules` to render 5 text cards and 2 charts is an engineering crime."**
