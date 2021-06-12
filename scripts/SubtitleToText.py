from os import listdir


def convert_subtitle_files_in_folder(folder, output_folder="", remove_new_lines=True):
    if output_folder == "":
        output_folder = folder

    all_file_names = listdir(folder)
    subtitle_file_paths = []
    out_file_paths = []
    for file_name in all_file_names:
        if file_name[-4:] == ".srt":
            out_file_name = file_name[:-4] + ".txt"
            subtitle_file_paths.append(folder + '/' + file_name)
            out_file_paths.append(output_folder + '/' + out_file_name)

    convert_subtitle_files(subtitle_file_paths, out_file_paths, remove_new_lines)


def convert_subtitle_files(files, out_files, remove_new_lines=True):
    for i in range(0, len(files)):
        convert_subtitle_file(files[i], out_files[i], remove_new_lines)


def convert_subtitle_file(subs_file, out_file, remove_new_lines=True):
    bad_words = ['-->', '</c>']

    contains_badwords = lambda line: any(bad_word in line for bad_word in bad_words)

    with open(subs_file) as oldfile, open(out_file, 'w') as newfile:
        lines = []
        prev_line = ""

        for line in oldfile:
            if not contains_badwords(line)\
                    and line != prev_line\
                    and not line.strip().isdecimal()\
                    and line != "\n"\
                    and line != " \n":
                prev_line = line
                if remove_new_lines:
                    lines.append(line.rstrip("\n") + " ")
                else:
                    lines.append(line)

        newfile.write("".join(lines))


def remove_new_lines_from_file(file, output_file=""):
    if output_file == "":
        output_file = file[:-4] + "_text-block" + ".txt"
    
    with open(file) as oldfile, open(output_file, 'w') as newfile:
        lines = []

        for line in oldfile:
            lines.append(line.rstrip("\n") + " ")

        newfile.write("".join(lines))
