

# maintains a mapping between a conflicted file in diff3 format and a conflicted file in default merge format

class Chunk:
    default_line_start = default_line_separator = default_line_end = 0
    diff3_line_start = diff3_line_separator = diff3_line_end = diff3_base_start = 0
    
    
    def __init__(self, default_line_start, diff3_line_start):
        self.default_line_start = default_line_start
        self.diff3_line_start = diff3_line_start
        self.base_content = []
    
    def set_end(self, default_line_end, diff3_line_end):
        self.default_line_end = default_line_end
        self.diff3_line_end = diff3_line_end
        

    def to_string(self):
        default = f"Default start: {self.default_line_start} \t Default separator: {self.default_line_separator} \t Default end: {self.default_line_end} \t"
        diff3 = f" | \t Diff3 start: {self.diff3_line_start} \t Diff3 separator: {self.diff3_line_separator} \t Diff3 end: {self.diff3_line_end} \n"
        
        return default + diff3

# default_file_path = "replay.txt"
# diff3_file_path = "replay-diff3.txt"



# file_default = []
# file_diff3 = []
# with open(default_file_path, 'r') as f:
#     file_default = f.readlines()
# with open(diff3_file_path, 'r') as f:
#     file_diff3 = f.readlines()

def get_chunks_mapping(file_default, file_diff3):

    CHUNK_START = "<<<<<<<"
    CHUNK_END = ">>>>>>>"
    CHUNK_SEP = "======="
    CHUNK_BASE_START = "|||||||"

    head_default = 0
    head_diff3 = 0

    chunks = []
    insideChunk = False
    left_chunk = False
    base_chunk = False
    right_chunk = False
    chunk = None
    while head_default < len(file_default):
        
        line_default = file_default[head_default]
        line_diff3 = file_diff3[head_diff3]

        if CHUNK_START in line_default:
            insideChunk = True
            left_chunk = True
            chunk = Chunk(head_default+1, head_diff3+1)
            head_default+=1
            head_diff3+=1
            continue
        if CHUNK_BASE_START in line_diff3:
            left_chunk = False
            base_chunk = True
            
            while CHUNK_SEP not in line_diff3:
                head_diff3+=1
                line_diff3 = file_diff3[head_diff3]
                chunk.base_content.append(line_diff3)
        if CHUNK_SEP in line_default:
            while CHUNK_SEP not in line_diff3:
                head_diff3+=1
                line_diff3 = file_diff3[head_diff3]
                if CHUNK_BASE_START not in line_diff3:
                    chunk.base_content.append(line_diff3)
            chunk.base_content.pop()
            base_chunk = False
            right_chunk = True
            chunk.default_line_separator = head_default+1
            chunk.diff3_line_separator = head_diff3+1
            head_default+=1
            head_diff3+=1
            continue
        if CHUNK_END in line_default:
            while CHUNK_END not in line_diff3:
                head_diff3+=1
                line_diff3 = file_diff3[head_diff3]
            insideChunk = False
            right_chunk = False
            chunk.set_end(head_default+1, head_diff3+1)
            chunks.append(chunk)
            chunk = None
            head_default+=1
            head_diff3+=1

            line_default = file_default[head_default]
            line_diff3 = file_diff3[head_diff3]

            while line_diff3 != line_default:
                head_default+=1
                line_default = file_default[head_default]
            continue

        head_default+=1
        head_diff3+=1
    
    #for chunk in chunks:
    #     print(chunk.to_string())
    #     print(chunk.base_content)
    return chunks