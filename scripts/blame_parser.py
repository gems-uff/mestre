import re
from datetime import datetime

pattern = re.compile(r"(\S+) (?:(\S+))?\s*\(<(.*?)>\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [\+|-]\d{4})\s+(\d+)\) (.*)")

class Blame:
    def __init__(self, file_lines):
        self.lines = []
        self.parse(file_lines)
    
    def parse(self, file_lines):
        global pattern
        for line in file_lines:
            x = pattern.match(line)
            if x is not None:
                commit_sha = x.group(1) # 9 characters sha
                file_path = x.group(2)
                author = x.group(3)
                commit_date = x.group(4)
                commit_date = datetime.strptime(commit_date, '%Y-%m-%d %H:%M:%S %z') 
                line_number = int(x.group(5))
                line_content = x.group(6)
                self.lines.append(BlameLine(commit_sha, file_path, author, commit_date, line_number, line_content))

    # find all lines that were modified before a given date and appears before the specified line
    # the returned list is ordered so that the first element is the closest line to the line_limit
    def find_lines_by_date_before_line_limit(self, date_limit, line_limit):
        date_limit = datetime.strptime(date_limit, '%Y-%m-%d %H:%M:%S %z')
        selected_lines = []
        for line in self.lines:
            if line.commit_date <= date_limit and line.line_number <= line_limit:
                selected_lines.append(line)
        return selected_lines[::-1]

    # find all lines that were modified before a given date and appears after the specified line
    # the returned list is ordered so that the first element is the closer line to the line_limit
    def find_lines_by_date_after_conflict_end_mark(self, date_limit, chunk_begin_line):
        selected_lines = []
        date_limit = datetime.strptime(date_limit, '%Y-%m-%d %H:%M:%S %z')
        start = False
        for line in self.lines:
            if line.line_number > chunk_begin_line and '>>>>>>>' in line.line_content:
                start = True
            if line.commit_date <= date_limit and start:
                selected_lines.append(line)
        return selected_lines

    def get_all_lines_content_before_date(self, date_limit):
        selected_lines = []
        date_limit = datetime.strptime(date_limit, '%Y-%m-%d %H:%M:%S %z')
        for line in self.lines:
            if line.commit_date <= date_limit:
                selected_lines.append(line.line_content)
        return selected_lines
            
class BlameLine:
    def __init__(self, commit_sha, file_path, author, commit_date, line_number, line_content):
        self.commit_sha = commit_sha
        self.file_path = file_path
        self.author = author
        self.commit_date = commit_date
        self.line_number = line_number
        self.line_content = line_content

def test():
    lines = []
    with open('test/blame.txt', 'r') as f:
        lines = f.readlines()

    blame = Blame(lines)
    base = "708e2db0bf5e3bfbb48bf94d604ef883970a2b92"
    # how to get the date: git show -s --format=%ci {SHA}
    base_date = "2015-09-18 09:53:31 -0700"

    # date = datetime.strptime(output, '%Y-%m-%d %H:%M:%S %z') 
    start = 20
    end = 32

    lines = blame.find_lines_by_date_after_line_limit(base_date, start)
    for line in lines:
        print(line.line_content)