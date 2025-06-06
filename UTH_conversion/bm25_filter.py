import sqlite3
from rank_bm25 import BM25Okapi

def get_domain_intent_keywords(domain):
    """Return relevant action keywords based on the website's primary purpose"""
    domain_keywords = {
        # E-commerce sites
        "amazon.com": ["buy", "add", "cart", "purchase", "order", "shop", "checkout", "wishlist"],
        "ebay.com": ["bid", "buy", "sell", "auction", "purchase", "cart", "order"],
        "etsy.com": ["buy", "purchase", "cart", "shop", "favorite", "order"],
        
        # Content/Media sites  
        "youtube.com": ["watch", "play", "subscribe", "like", "comment", "share", "upload"],
        "netflix.com": ["watch", "play", "stream", "episode", "movie", "series", "resume"],
        "spotify.com": ["play", "listen", "playlist", "song", "album", "follow", "shuffle"],
        
        # Productivity sites
        "docs.google.com": ["write", "edit", "format", "document", "text", "share", "comment"],
        "sheets.google.com": ["calculate", "formula", "cell", "chart", "data", "filter", "sort"],
        "notion.so": ["write", "edit", "page", "block", "template", "database", "organize"],
        
        # Social media
        "twitter.com": ["tweet", "post", "share", "follow", "like", "retweet", "comment"],
        "facebook.com": ["post", "share", "like", "comment", "friend", "message", "photo"],
        "linkedin.com": ["connect", "message", "share", "post", "endorse", "follow", "apply"],
        
        # Development
        "github.com": ["commit", "push", "pull", "code", "repository", "fork", "clone", "merge"],
        "stackoverflow.com": ["ask", "answer", "vote", "comment", "search", "question", "solution"],
        
        # Search/Information
        "google.com": ["search", "find", "query", "result", "explore", "discover"],
        "wikipedia.org": ["read", "search", "article", "edit", "reference", "learn"],
        
        # Banking/Finance
        "chase.com": ["transfer", "pay", "deposit", "balance", "statement", "account"],
        "paypal.com": ["send", "pay", "transfer", "receive", "balance", "transaction"],
    }
    
    # Extract main domain (remove subdomains)
    main_domain = domain.lower()
    if main_domain.startswith('www.'):
        main_domain = main_domain[4:]
    
    # Return specific keywords or default to general action words
    return domain_keywords.get(main_domain, ["click", "select", "view", "navigate", "action", "complete"])

def bm25_filter(db_path="intents.db", threshold=0.3):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all actions with their domains
    rows = cur.execute("""
        SELECT id, domain, action_text FROM intents WHERE action_text IS NOT NULL AND domain IS NOT NULL
    """).fetchall()
    
    if not rows:
        print("No actions found in database")
        return
    
    # Group actions by domain for more efficient processing
    domain_groups = {}
    for row_id, domain, action_text in rows:
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append((row_id, action_text))
    
    print(f"Processing {len(domain_groups)} domains...")
    
    # Process each domain with its specific intent keywords
    for domain, actions in domain_groups.items():
        print(f"Scoring {len(actions)} actions for {domain}")
        
        # Get domain-specific keywords
        query_terms = get_domain_intent_keywords(domain)
        print(f"  Using keywords: {query_terms}")
        
        # Prepare BM25 for this domain
        action_texts = [action[1] for action in actions]
        tokenized_actions = [action.lower().split() for action in action_texts]
        
        if not tokenized_actions:
            continue
            
        bm25 = BM25Okapi(tokenized_actions)
        
        # Score each action against domain-specific terms
        scores = bm25.get_scores([term.lower() for term in query_terms])
        
        # Update database with scores
        for i, (action_id, action_text) in enumerate(actions):
            score = float(scores[i]) if i < len(scores) else 0.0
            
            cur.execute("""
                UPDATE intents SET bm25_score = ? WHERE id = ?
            """, (score, action_id))
    
    conn.commit()
    conn.close()
    
    print(f"Updated BM25 scores for {len(rows)} actions across {len(domain_groups)} domains")

