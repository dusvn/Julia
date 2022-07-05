function radixSort!(A)
    i=maximum(A) 
    exp=1
    while i/exp>0 
       insertionSort!(A,exp)
        exp*=10   
    end 
end
 
function insertionSort!(A,exp)
        for j in 2:length(A)
            key = A[j]
            i=j-1 
            while(i>0 && (A[i]÷exp)%10 > (key÷exp)%10)  
                A[i+1] = A[i]
                i-=1 
            end 
            A[i+1]=key 
        end 
end 