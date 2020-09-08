import json
from itertools import combinations

from Extraction.extract import Extract
from Extraction.paper import PaperStruct, PageStruct, Article
import math
from os.path import basename
from sys import maxsize
import re


class Construct:
    def construct(self, path):
        e = Extract(path)
        files = e.run()
        print(len(files))
        for file in files:
            self.get_articles(file)
    
    def get_articles(self, file):
        """
        Takes a file and generates articles.
        :param file: A file containing blocks with textlines.
        :return: A list of articles.
        """
        # Sort first by distance to top-right corner, then by vertical position, to ensure a headline is added before
        # a subheadline, titles are added from right to left.
        file.txtblocks.sort(key = lambda e: (e.hpos + e.vpos, e.vpos))

        arts = self.find_headlines(file)

        # Sort both file and arts by horizontal position then by vertical,
        # as the body of an article is structured this way.
        file.txtblocks.sort(key = lambda e: (e.hpos, e.vpos))
        arts.sort(key = lambda e: (e.hpos, e.vpos))

        no_match_blocks = []

        # Add blocks to articles
        for block in file.txtblocks:
            tmpart = None

            # Finds the closest article.
            for article in arts:
                if (self.is_body_of_article(article, block)):
                    if (self.horizontal_distance(article, block) < self.horizontal_distance(tmpart, block)):
                        tmpart = article

            if tmpart != None:
                for line in block.txtlines:
                    tmpart.body.append(line)
            else:
                no_match_blocks.append(block)

        #print(f"File: {file.original_filepath}, number of unmatched: {len(no_match_blocks)}")

        if len(no_match_blocks) < 100:
            self.write_articles(arts, file)
        else:
            print(f"WARNING:File {file.original_filepath} skipped. To many unmatched textblocks, most likely irrelevant.")

        return arts

    def group_unmatched(self, blocks):
        print(f"Number of unmatched blocks {len(blocks)}")

        # Create pairs of
        pairs = []
        for b in blocks:
            for b2 in blocks:
                distance = math.sqrt(pow(b.hpos - b2.hpos, 2) + pow(b.vpos - b2.vpos, 2))

                if self.is_below(b, b2):
                    if (b, b2) in pairs or (b2, b) in pairs:
                        continue
                    pairs.append((b, b2))

        groups = []
        for b in blocks:
            g = self.make_group(b, pairs, set([b]))

            for block in g:
                if block in blocks:
                    blocks.remove(block)
                for p in pairs:
                    if block in p:
                        pairs.remove(p)

            if g:
                groups.append(g)

        return groups

    def make_group(self, block, pairs, group, index=0):
        if len(pairs) <= index:
            for p in pairs:
                if block in p:
                    pairs.remove(p)
            return group

        pair = pairs[index]
        b = None
        if block in pair:
            if pair[0] == block:
                b = pair[1]
            else:
                b = pair[0]

            group.add(b)

        g = self.make_group(block, pairs, group, index + 1)

        if b != None:
            g.union(self.make_group(b, pairs, group))

        return g

    def is_below(self, b1, b2):
        return (abs(b1.hpos - b2.hpos) <= 100
                and abs(b1.vpos - b2.vpos + b2.height) <= 500)

    def find_headlines(self, file):
        """
        Find headlines given a file
        :param file: Find headlines by checking the size.
        :return: A list of articles with a position and headline.
        """
        articles = []

        # Find headlines
        for block in file.txtblocks[:]:
            if block.txtlines[0].style > 15:
                art = self.make_article(block)
                no_parent = True

                # Checks if subtitle by checking for headline above.
                for article in articles:
                    if self.is_subheadline(article, block):
                        article.subheadline += ' ' + art.headline if article.subheadline else art.headline
                        no_parent = False
                        break

                if no_parent:
                    articles.append(art)

                file.txtblocks.remove(block)

        return articles

    def is_subheadline(self, article_block, block):
        """
        Checks if block is a contains text that is a subheadline of article or not.
        :param article_block: An article with headline
        :param block: A block containing text with a size equal to that of a headline.
        :return: True if subheadline of article; otherwise False.
        """
        return (abs((article_block.vpos + article_block.height) - block.vpos) < 500
                    and abs(article_block.hpos - block.hpos) < 150)

    def is_body_of_article(self, article, block):
        return (block.hpos >= article.hpos - 100            # Block is to the right of the beginning of the article.
                and block.vpos > article.vpos               # Block is below the start of the article.
                and block.txtlines[0].style < article.style     # Text size is smaller than the article headline.
                and (block.hpos) <= (article.hpos + article.width)) # Text starts beneath the headline

    def is_next_text_body(self, article_block, block):
        leftmost_point = article_block.hpos + article_block.width
        lowest_point = article_block.vpos + article_block.height
        horizontal_dist = abs(block.hpos - leftmost_point)
        vertical_dist = abs(block.vpos - lowest_point)

        if (block.hpos > leftmost_point
                and horizontal_dist <= 500):
            return True
        elif(block.vpos > lowest_point
                and vertical_dist <= 10000):
            return True
        else:
            return False

    def make_article(self, block):
        """
        Makes an article given a block.
        :param block: A block of Block type.
        :return: Returns a article with position, size, style, and headline is defined.
        """
        art = Article(block.id, block.hpos, block.vpos, block.width, block.height, block.txtlines[0].style, block)
        for line in block.txtlines:
            if line.style > 15:
                art.headline = art.headline + ' ' + (' '.join(line.words)) if art.headline else (' '.join(line.words))

        return art

    def horizontal_distance(self, article, block):
        if article == None or block == None:
            return maxsize
        else:
            return abs(article.vpos - block.vpos)

    def write_articles(self, articles, file):
        filename = file.original_filepath
        picpath = file.original_picpath
        # with open(re.sub('.alto.xml', '.txt', filename), "w") as f:
        #     for article in articles:
        #         f.write(article.headline)
        #         f.write(article.subheadline)
        #         f.write('\n')
        #         f.write(' '.join([' '.join(line.words) for line in article.body]))
        #         f.write(' \n\n')

        with open(re.sub('.alto.xml', '.json', filename), "w") as f:
            json.dump(self.create_dict(articles, filename, picpath), f)

    def create_dict(self, articles, name, path):
        date = re.search('.+-([0-9]{4}-[0-9]{2}-[0-9]{2})', name).group(1)
        dicts = []
        for article in articles:
            if not article.body:
                continue
            elif self.skip_article(article):
                continue


            dicts.append({'id':  date + article.get_id(),
                          'headline': article.headline,
                          'subheadline': article.subheadline,
                          'body': ' '.join([' '.join(line.words) for line in article.body]),
                          'pic_path': path,
                          'thedate': date})

        return dicts


    def skip_article(self, article):
        # Article to short. Most likely advertisements
        if (sum([len(line.words) for line in article.body]) < 100):
            return True
        elif self.get_percentage_numbers(article):
            print(f"it worked {article.get_id()}")
            return True

    def get_percentage_numbers(self, article):
        chars = [char for line in article.body for word in line.words for char in word]
        lenght = len(chars)

        num_digits = 0
        for c in chars:
            if c.isdigit():
                num_digits += 1

        return num_digits >= 50

