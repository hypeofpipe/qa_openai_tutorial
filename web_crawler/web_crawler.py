from requests_html import HTMLSession
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import gzip
import os

# REGEX pattern to match URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

domain = 'zakon.rada.gov.ua'
full_url = 'https://zakon.rada.gov.ua/laws/show/3543-12#Text'

# Create a class to parse the HTML and get hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # List to store hyperlinks
        self.hyperlinks=[]
    # Override the handle_starttag method to get hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        # If the tag is an anchor tag and has an href attribute, add to hyperlinks list
        if tag == 'a' and 'href' in attrs:
            self.hyperlinks.append(attrs['href'])
    # Get hyperlinks from URL
    def get_hyperlinks(self, url):
        try:
            with urllib.request.urlopen(url) as response:
                if not response.info().get_content_type() == 'text/html':
                    return []
                if response.info().get('Content-Encoding') == 'gzip':
                    html = gzip.decompress(response.read()).decode('utf-8')
                else:
                    html = response.read().decode(response.headers['Content-Encoding'])
        except Exception as e:
            print(e)
            return []

        parser = HyperlinkParser()
        parser.feed(html)
        
        return parser.hyperlinks
    # Get hyperlinks of the same domain
    def get_domain_hyperlinks(self, local_domain, url):
        clean_links = []
        for link in set(self.get_hyperlinks(url)):
            clean_link = None
            # If a link is URL, check if it's within the same domain
            if re.search(HTTP_URL_PATTERN, link):
                # Parse URL and check domain
                url_obj = urlparse(link)
                if url_obj.netloc == local_domain:
                    clean_links.append(link)
            # If a link is not URL, check if it's a relative path
            else:
                if link.startswith('/'): 
                    link = link[1:]
                elif link.startswith('#') or link.startswith('mailto:'):
                    continue
                clean_link = 'https://' + local_domain + '/' + link
                
            if clean_link is not None:
                if clean_link.endswith('/'):
                    clean_link = clean_link[:-1]
                clean_links.append(clean_link)
                
            # Return the list of hyperlinks within the same domain
            return list(set(clean_links))
    def crawl(self, url):
        local_domain = urlparse(url).netloc
        
        # queue to store URLs to crawl
        queue = deque([url])
        # set URLs that already have been seen, no duplicates
        seen = set([url])
        
        current_directory = os.getcwd()
        
        if not os.path.exists(os.path.join(current_directory, 'text')):
            os.mkdir(os.path.join(current_directory, 'text'))
            
        if not os.path.exists(os.path.join(current_directory, 'text' + local_domain + '/')):
            os.mkdir(os.path.join(current_directory, 'text' + local_domain + '/'))
            
        if not os.path.exists(os.path.join(current_directory, 'processed')):
            os.mkdir(os.path.join(current_directory, 'processed'))
        
        # while queue is not empty, crawl
        while queue:
            next_url = queue.pop()
            
            # In order to render JavaScript, we need to use HTMLSession
            session = HTMLSession()
            response = session.get(url)
            response.html.render()
            
            encoding = response.encoding
            
            print(next_url, response.headers['content-encoding'], encoding)
            
            # Get HTML of the page with content directly
            text = response.html.html
            
            # save text from url to <url>.txt
            with open('text' + local_domain + '/' + url[8:].replace('/', '_') + '.txt', 'w', encoding=encoding) as f:
                # get text from URL
                soup = BeautifulSoup(text, 'html.parser')
                
                # remove the tags from text
                text = soup.get_text()            
                
                # stop the crawl if we need javascript
                if ("You need to enable JavaScript to run this app." in text):
                    print('Unable to parse page: ' + next_url + ' due to JavaScript being required')
                
                f.write(text)
        
            ## TO-DO: check if it works
            # get hyperlinks from the URL    
            domain_hyperlinks = self.get_domain_hyperlinks(local_domain, next_url)
            
            if (domain_hyperlinks is None):
                continue
            
            for link in self.get_domain_hyperlinks(local_domain, next_url):
                if link not in seen:
                    queue.append(link)
                    seen.add(link)
        
parser = HyperlinkParser()


parser.crawl(full_url)
