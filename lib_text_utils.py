# coding=utf-8

import codecs
# import chardet

def write_to_file(file_incl_path, line_list):
  """
  Writes the lines contained in line_list to a file
  """
  #outFile = open(file_incl_path, 'w')
  outFile = codecs.open(file_incl_path, "w", "utf-8")
  for line in line_list:
    #print line  
    outFile.write(line + "\n")
  outFile.close()


def is_ext_ascii(instr):
  """
  Returns true if the input is in extended ASCII, false otherwise
  """
  return all(ord(c) < 256 for c in instr)
  """
  retval = False
  try:
    encoding = chardet.detect(instr)
    print instr, encoding
    if encoding['encoding'] == "ascii":
      retval = True
  except:
    retval = False
  return retval
  """

def latex_prepend_index_nos(strarr):
  """
  Prepends index numbers to the elements of an array of strings
  """
  arr_nos = range(1, len(strarr)+1)
  return ["$^" + "{" + str(n) + "}$" + k for (n, k) in zip(arr_nos, strarr)]

