a=1
for i in faces/*; do
	new=$(printf "%05d" "$a")
	mv -i -- "$i" "$new"
	let a=a+1
 done
