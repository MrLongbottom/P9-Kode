class FileStruct():
    def __init__(self, name, orig):
        self.original_filepath = name
        self.original_picpath = orig
        self.txtblocks = []
    
    def add_txtblock(self, block):
        self.txtblocks.append(block)

    def __str__(self):
        return self.original_filepath

class TextBlock:
    def __init__(self, id, height, width, hpos, vpos):
        self.id = id
        self.height = height
        self.width = width
        self.hpos = hpos
        self.vpos = vpos
        self.txtlines = []

    def add_line(self, line):
        self.txtlines.append(line)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.id)

class TextLine:
    def __init__(self, id, style, height, width, hpos, vpos):
        self.id = id
        self.words = []
        self.style = style
        self.height = height
        self.width = width
        self.hpos = hpos
        self.vpos = vpos

    def add_word(self, word):
        self.words.append(word)

    def __str__(self):
        dict_ = {"ID": self.id, "STYLEREFS": self.style, "HEIGHT": self.height, "WIDTH": self.width, "HPOS": self.hpos, "VPOS": self.vpos, "WORDS": self.words}
        return str(dict_)