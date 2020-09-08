class PaperStruct:
    _published = None
    _paper = None
    _pages = []

class PageStruct:
    _pagenum = -1
    _topic = None
    _articles = {}

    def __init__(self, filepath, picpath):
        self._original_filepath = filepath
        self._original_picpath = picpath

    def set_original_pic_path(self, pic_path):
        self._original_picpath = pic_path

    def add_article(self, key, article):
        if key in self._articles:
            self._articles[key].append(article)
        else:
            self._articles[key] = [article]

    def set_page_num(self, pagenum):
        self._pagenum = pagenum

    def __str__(self):
        return str(self._articles)

class Article:

    def __init__(self, id, hpos, vpos, width, height, styleval, block):
        self._id = id
        self.style = styleval
        self.hpos = hpos
        self.vpos = vpos
        self.width = width
        self.height = height

        self.last_added_block = block
        self.headline = ""
        self.subheadline = ""
        self.author = ""
        self.body = []

    def get_id(self):
        return self._id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        dict_ = {"id": self._id}#, "headline" : self.headline, "author" : self.author}#, "body" : self.body}
        return str(dict_)