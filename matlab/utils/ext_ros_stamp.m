function t = ext_ros_stamp(stampedmsg)
sec = stampedmsg.Header.Stamp.Sec;
nsec = stampedmsg.Header.Stamp.Nsec;
t = str2double(append(num2str(sec) ,'.' ,num2str(nsec)));
end
