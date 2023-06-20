sec_per_frame = 0.0125

time_stamp_file = "data/demo/vad_testdata_xuhui.txt"
output_label_file = "data/demo/vad_testdata_xuhui.label"
fout = open(output_label_file, 'w')

with open(time_stamp_file, 'r') as fin:
    cur_time = 0.0
    for line in fin:
        line = line.strip().split('\t')
        start = line[0]
        end = line[1]
        label = line[2]
        start = start.split(":")
        end = end.split(":")
        assert len(start)<=3, len(end)<=3
        start_sec = 0.0
        for i in range(len(start)):
            start_sec += float(start[-(1+i)]) * 60 ** i
        end_sec = 0.0
        for i in range(len(end)):
            end_sec += float(end[-(1+i)]) * 60 ** i
        label = 1 if label == "active" else 0
        while cur_time <= end_sec :
            fout.write(str(label)+" ")
            cur_time += sec_per_frame

