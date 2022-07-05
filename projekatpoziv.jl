include("projekat.jl")
A=[9 7 8 5 64 2 3 11 6 4 12 41 42 44 45 46 47 48 51 52 55 54 98 77 76 67 17 29 30 88]

println("Izgled niza pre sortiranja radix sort metodom")
println()
display(A)
radixSort!(A)
println()
println("Izgled niza nakon sortiranja radix sort metodom")
println()
display(A)
