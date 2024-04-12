from requests_html import HTMLSession
import re
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import gzip
import os

# REGEX pattern to match URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

domain = 'zakon.rada.gov.ua'
full_url = 'https://zakon.rada.gov.ua/laws/show/3621-20'

# Create a class to parse the HTML and get hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # List to store hyperlinks
        self.hyperlinks=[]
        self.current_directory = os.getcwd() + '/web_crawler/'
    # Override the handle_starttag method to get hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        # If the tag is an anchor tag and has an href attribute, add to hyperlinks list
        if tag == 'a' and 'href' in attrs:
            self.hyperlinks.append(attrs['href'])
    # Get hyperlinks from URL
    def get_hyperlinks(self, url):
        try:
            ## It's better to use requests_html to render JavaScript
            session = HTMLSession()
            response = session.get(url)
            with session.get(url) as response:
                response.html.render(scrolldown=1,sleep=5, wait=5)
        except Exception as e:
            print(e)
            return []
        
        response
        
        return response.html.links
    # Get hyperlinks of the same domain
    def get_domain_hyperlinks(self, local_domain, url):
        # Regex patterns to ignore paths that start with specific patterns
        regex_patterns_to_ignore = [
            r'^/laws/main/',
            r'^/laws/show/privacy',
            r'^/laws/card'
        ]
        compiled_ignore_regex = [re.compile(pattern) for pattern in regex_patterns_to_ignore]

        clean_links = []
        for link in set(self.get_hyperlinks(url)):
            if not re.match(HTTP_URL_PATTERN, link):
                # For non-HTTP links, you may want to filter out or handle differently as you wish
                continue

            url_obj = urlparse(link)
            if url_obj.netloc != local_domain:
                continue

            # Check if any of the ignore patterns match
            if any(regex.match(url_obj.path) for regex in compiled_ignore_regex):
                print('Ignoring path: ' + link)
                continue

            # Normalizing relative path
            clean_link = None
            if url_obj.path.startswith('/'):
                clean_link = 'https://' + local_domain + url_obj.path
            else:
                clean_link = 'https://' + local_domain + '/' + url_obj.path

            if clean_link and clean_link.endswith('/'):
                clean_link = clean_link[:-1]

            clean_links.append(clean_link)
        
        return list(set(clean_links))  # Return deduplicated list
    def crawl(self, url):
        
        local_domain = urlparse(url).netloc
        
        # queue to store URLs to crawl
        queue = deque([url])
        long_queue = deque([])
        # set URLs that already have been seen, no duplicates
        seen = set([url])
            
        if not os.path.exists(os.path.join(self.current_directory, 'text' + local_domain + '/')):
            os.mkdir(os.path.join(self.current_directory, 'text' + local_domain + '/'))
        
        # while queue is not empty, crawl
        while queue:
            next_url = queue.pop()
            url_for_print = next_url + '/print'
            
            # In order to render JavaScript, we need to use HTMLSession
            session = HTMLSession()
            response = session.get(url_for_print)
            response.html.render(sleep=5, wait=5, scrolldown=1)
            
            if (response.text.find('Відбувається форматування тексту') != -1):
                print('Page ' + url_for_print + ' is still loading on the server, retrying later')
                long_queue.append(next_url)
                continue
            
            encoding = response.encoding
            
            print(url_for_print, encoding)
            
            # Get HTML of the page with content directly
            text = response.text
            
            # save text from url to <url>.txt
            try:
                with open(self.current_directory + 'text' + local_domain + '/' + url_for_print[8:].replace('/', '_') + '.txt', 'w', encoding=encoding) as f:
                    # get text from URL
                    soup = BeautifulSoup(text, 'html.parser')
                    
                    # remove the tags from text
                    text = soup.get_text()            
                    
                    # stop the crawl if we need javascript
                    if ("You need to enable JavaScript to run this app." in text):
                        print('Unable to parse page: ' + url_for_print + ' due to JavaScript being required')
                    
                    f.write(text)
            except Exception as e:
                print(e)
                continue
        
            ## TO-DO: check if it works
            # get hyperlinks from the URL    
            domain_hyperlinks = self.get_domain_hyperlinks(local_domain, next_url)
            
            if (domain_hyperlinks is None):
                continue
            
            for link in domain_hyperlinks:
                if link not in seen:
                    queue.append(link)
                    seen.add(link)
        # refactor the code to crawl the long_queue
        while long_queue:
            next_url = long_queue.pop()
            url_for_print = next_url + '/print'
            
            # In order to render JavaScript, we need to use HTMLSession
            session = HTMLSession()
            response = session.get(url_for_print)
            response.html.render(sleep=10, wait=60, scrolldown=1)
            
            if (response.text.find('Відбувається форматування тексту') != -1):
                print("Page ' + url_for_print + ' is still loading on the server, we're giving up!")
                continue
            
            encoding = response.encoding
            
            print(url_for_print, encoding)
            
            # Get HTML of the page with content directly
            text = response.text
            
            # save text from url to <url>.txt
            try:
                with open(self.current_directory + 'text' + local_domain + '/' + url_for_print[8:].replace('/', '_') + '.txt', 'w', encoding=encoding) as f:
                    # get text from URL
                    soup = BeautifulSoup(text, 'html.parser')
                    
                    # remove the tags from text
                    text = soup.get_text()            
                    
                    # stop the crawl if we need javascript
                    if ("You need to enable JavaScript to run this app." in text):
                        print('Unable to parse page: ' + url_for_print + ' due to JavaScript being required')
                    
                    f.write(text)
            except Exception as e:
                print(e)
                continue
        
            ## TO-DO: check if it works
            # get hyperlinks from the URL    
            domain_hyperlinks = self.get_domain_hyperlinks(local_domain, next_url)
            
            if (domain_hyperlinks is None):
                continue
            
            for link in domain_hyperlinks:
                if link not in seen:
                    long_queue.append(link)
                    seen.add(link)
        
parser = HyperlinkParser()


parser.crawl(full_url)
