#DUSAN PARIPOVIC PR76/2020 
#PROJEKAT PA GRUPA 4 

using Flux
using DataFrames
using CSV
using StatsBase
using Lathe.preprocess: TrainTestSplit

## 1. Neuronska mreža u biblioteci Flux

# Ovde je opisna neuronska mreža sa jednim slojem i 
# linearnom aktivacionom funkijom σ(x) = x
# U našem slučaju sa kreditima mreža ima 3 ulaza (features ili inputs)
# To su x = PRIHOD_PODNOSIOCA, MESECNA_RATA, PERIOD_ZA_VRACANJE_MESECI
# Mreža ima 1 izlaz (label ili output)
# To je y = IZNOS_KREDITA


# Ta se neuronska mreža može modelovati
# pomoću Flux funkcije Dense.
model = Dense(3, 1)  # 3 inputa, 1 output 
# Dense(n, m, σ)
# Dense postavlja početne vrednosti slobonih članova kao slučajne,

# 3.0 ucitavanje podataka iz CSV Fajla
data = DataFrame(CSV.File("projekatPA.CSV")) 

println("[IZGLED DATA FRAMEA PRE SREDJIVANJA]") 
println("-----------------------------------")
display(describe(data))
println()

## 3.1 Sređivanje (čišćenje) podatka
# ubacivanje prosecne mesecne rate na mesto gde fali 
data[ismissing.(data[!, :MESECNA_RATA]),:MESECNA_RATA] .= trunc(Int64,mean(skipmissing(data[!,:MESECNA_RATA]))) 

# ubacivanje prosecnog perioda za vracanje kredita u mesecima tamo gde fali
data[ismissing.(data[!, :PERIOD_ZA_VRACANJE_MESECI]),:PERIOD_ZA_VRACANJE_MESECI] .= trunc(Int64,mean(skipmissing(data[!,:PERIOD_ZA_VRACANJE_MESECI]))) 

println("[IZGLED DATA FRAMEA POSLE SREDJIVANJA]") 
println("-----------------------------------")
display(describe(data)) 
println()



# 3.2 filtriranje
filter!(row->row.IZNOS_KREDITA>250 && row.IZNOS_KREDITA<5000,data)         #zanemarujem iznos kredita manji od 250e i vece od 5k eura 
                                                                           #250e u smislu podizanja kredita za neki uredjaj odlozeno placanje itd
filter!(row->row.PRIHOD_PODNOSIOCA>250 && row.PRIHOD_PODNOSIOCA<5000,data) #plata 250+ zbog vracanja(minimalac) 
                                                                           #manja od 5k zbog greske 

## 2. Implemetacija (modelovanje)

# 2.1 Podeliti ulazne podatke na trening i test 
data_train, data_test = TrainTestSplit(data, 0.8)   #0.8==80% se odvaja za treniranje ostalo su testni podaci 

# 2.2 

#konverzija da bih mogao sve da ih ubacim u novi dataset
x_train = convert(Array{Float64}, select(data_train, Not([:IZNOS_KREDITA])))'
y_train = convert(Array{Float64}, select(data_train, :IZNOS_KREDITA))'
test    = convert(Array{Float64}, select(data_test, Not([:IZNOS_KREDITA])))'
# 2.3 Treniranje modela pomoću neuronske mreže
# x je niz od 3 elementa (PRIHOD_PODNOSIOCA, MESECNA_RATA, PERIOD_ZA_VRACANJE_MESECI), y je skalar(IZNOS_KREDITA)

loss(x, y) = Flux.mse(model(x), y) #default definicija f-je 

par = params(model)            #ovo radimo zato sto trening f ne moze da se posalje baš model nego moramo da konvertujemo u params
                               #menjamo samo format podataka

opt = Flux.Descent(0.00000001) #Gradijentni spust, argument je brzina učenja
                               #Izbor stope učenja teba pažljivo izabrati 
                               #velika stopa može ovesti do divergencije modela.
                               #U pozdani se odvija trazenje parcijalnog izvoda f-je 
                               #trazenje ekstremnih vrednosti 

dataset= [(x_train,y_train)]

for step in 1:500                                #1 step je jedna epoha, najmanju gresku dobijam za 500 epoha MAPE=32-3%
    println("TRENING JE U PROCESU SACEKAJ!!! ")   #u jednoj epohi se model trenira onoliko puta koliko ima vrsta u fajlu[posmatramo fajl kao matricu]
    Flux.train!(loss, par, dataset, opt)
end
println()
println("[MODEL JE SPREMAN]")

# model je spreman
println("-----[PRIKAZ MODELA]-----")
println(model.W, model.b) #.W su težine grana ,.b ->bias ili ti slobodan član u f-ji 
                          #kada pravimo model to je ustv objekat i f-ja u isto vreme[kada gledamo iz uglav Objktnog programiranja] 
println()
println(params(model))    #prikaz svega zajedno 
println()
println()
println()
## 4. Računanje grešaka - modul Flux.Losses

# Biblioteka Flux ima veliki broj statističkih funkcija za pa tako i za greške
# One se mogu promaći u modulu Flux.Losses
# oznake koje ćemo oristiti

y_model = model(x_train)
y_model1 = model(test)

errors_train= y_train - y_model  # gresku u treniranju dobijamo kao trening podaci - predikt podaci tj ono sto je model uradio

mae = mean(abs.(errors_train))  # MAE - Mean absolute error
                                # (srednja apsolutna greška, ili
                                # prosek apsolutnih vrednosti)
                                # mae(ŷ, y; agg = mean)
                                
                                
mse = mean(abs.(errors_train.*errors_train))    # MSE - Mean Squared Error
                                                # (Srednja kvadratna greška, ili
                                                # prosek kvadrata greške)


rmse = sqrt(mse)                # RMSE - Root Mean Squared Error
                                # (koren srednje kvadratne greške, ili
                                # koren proseka kvadrata greške)



# MAPE - Mean Absolute Percentage Error
# (Prosecna relativna greska)
                                                 #y_model je prediktovana vr 
mape = mean(abs.(y_model-y_train)./y_train)*100  # mape greska je u procentima zbog toga je *100 
                                                 #prediktovana - stvarna vrednost(trenirana) / stvarna(trenirana)

println("[------GRESKE-----]")
println("1.MSE=$mse")  
println("------------")
println("2.RMSE=$rmse")
println("------------")
println("3.MAE=$rmse")
println("------------")
println("4.MAPE=$mape%")
println("------------")
