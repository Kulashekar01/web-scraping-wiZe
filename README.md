# web-scraping-wiZe
A web scraping pipeline that extracts all articles written by a specific author, collects relevant blog posts, cleans the scraped data, and classifies the articles into different relevant topics.

Here’s a structured step-by-step approach to guide you through completing this task:

### Step 1: Setting Up the Environment
1. **Install Necessary Libraries**:
   - Ensure Python is installed, and create a virtual environment for your project.
   - Install required libraries:  
     ```bash
     pip install requests beautifulsoup4 scrapy selenium pandas nltk scikit-learn matplotlib seaborn
     ```
     - **Requests** or **Scrapy** for web scraping
     - **BeautifulSoup4** for HTML parsing
     - **Selenium** if dynamic loading is needed
     - **Pandas** for data manipulation
     - **NLTK** or **Scikit-learn** for topic classification

### Step 2: Web Scraping
1. **Identify Target Website(s)**:
   - Choose the main website (e.g., Medium) or several websites with articles from your target author.
   - Verify if they allow scraping by checking the site’s `robots.txt`.

2. **Build Scraper**:
   - Start with **Scrapy** or **Requests + BeautifulSoup** for static pages. If the site uses JavaScript to load content dynamically, use **Selenium**.
   - **Extract Essential Information**:
     - Title, Author, Date, Content Body, Tags/Topics if available.
   - **Handle Pagination**:
     - Look for pagination elements to navigate across multiple pages and collect all articles.
   - **Example Scrapy Code**:
     ```python
     import scrapy
     
     class ArticleSpider(scrapy.Spider):
         name = 'article_spider'
         start_urls = ['<target-url>']
         
         def parse(self, response):
             for article in response.css('div.article'):
                 yield {
                     'title': article.css('h2::text').get(),
                     'author': article.css('.author-name::text').get(),
                     'date': article.css('.publication-date::text').get(),
                     'content': ' '.join(article.css('p::text').getall())
                 }
             next_page = response.css('a.next-page::attr(href)').get()
             if next_page:
                 yield response.follow(next_page, self.parse)
     ```

3. **Dynamic Content Handling** (if needed):
   - Use **Selenium** to load the page, scroll through it, and extract the same elements as above.

4. **Save Data**:
   - Save the extracted articles in **CSV** or **JSON** format, including metadata.

### Step 3: Data Cleaning
1. **Data Parsing**:
   - Load the data using **Pandas** and perform initial inspection.
     ```python
     import pandas as pd
     data = pd.read_csv('articles.csv')
     ```

2. **Clean HTML & Metadata**:
   - Remove any unnecessary HTML tags, spaces, and special characters.
   - Remove comments, advertisements, and any duplicate entries.

3. **Normalize Text**:
   - Convert text to lowercase, remove punctuation, numbers, and stopwords to standardize the content.
     ```python
     import re
     from nltk.corpus import stopwords
     
     def clean_text(text):
         text = re.sub(r'<.*?>', '', text)
         text = re.sub(r'[^a-zA-Z\s]', '', text)
         text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
         return text.lower()
     
     data['cleaned_content'] = data['content'].apply(clean_text)
     ```

### Step 4: Topic Classification
1. **Choose Classification Approach**:
   - For simplicity, start with **keyword-based** classification. Alternatively, use **TF-IDF** with a clustering algorithm like **LDA** for unsupervised topic discovery or train a supervised model if labeled data is available.

2. **Rule-Based Classification (if applicable)**:
   - Define keywords for each topic, and classify articles based on keyword frequency.

3. **Machine Learning Approach**:
   - Use **TF-IDF** to convert article text to vectors.
   - Apply a classifier (Naive Bayes or SVM) for supervised learning or **LDA** for unsupervised classification.
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.decomposition import LatentDirichletAllocation
   
   tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
   tfidf = tfidf_vectorizer.fit_transform(data['cleaned_content'])
   
   lda = LatentDirichletAllocation(n_components=5, random_state=0)
   data['topic'] = lda.fit_transform(tfidf).argmax(axis=1)
   ```

4. **Evaluate and Fine-tune**:
   - If using a machine learning model, split data into training and testing sets, and assess classification accuracy. Aim for >70%.

5. **Topic Visualization (Bonus)**:
   - Use **Matplotlib** or **Seaborn** for topic visualization, such as word clouds or topic clusters.

### Step 5: Ethical Considerations
1. **Adhere to Robots.txt and Rate-Limiting**:
   - Implement delays between requests and rotate the user agent to avoid detection.
   - Respect the website’s terms of service.

### Step 6: Final Deliverables and Documentation
1. **Write Code for Reusability and Modularity**:
   - Separate scraping, cleaning, and classification code into different functions or files.
   
2. **ReadMe Documentation**:
   - Explain how to run each component.
   - Describe dependencies and installation steps.
   - Include guidelines for re-running the topic classification or adjusting parameters.

### Final Checklist:
- Ensure the **scraper** fetches all relevant articles and handles pagination.
- Verify the **cleaned data** is accurate and free of unwanted content.
- Check **classification accuracy** and improve if needed.
- Review for **adherence to ethical guidelines** in web scraping practices.
