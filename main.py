import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Se vuoi disattivare oneDNN
from encoding import Encoding
import random
from hamiltonian import QuantumEvolver
import gensim.downloader as api
from qiskit.quantum_info import partial_trace
from qiskit.visualization import circuit_drawer
from qiskit.visualization import plot_histogram
import time
import math
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Operator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import defaultdict
from qiskit import ClassicalRegister
from collections import Counter
from scipy.linalg import qr
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from PIL import Image


from collections import defaultdict
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate
from layer import ansatzBuilder
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import HGate
from datasets import load_dataset
from collections import Counter
import numpy as np
import os

def showCircuitFromStateVector(statevector):
    """
    Visualizza il circuito che inizializza il dato statevector.
    """
    qubits = int(np.log2(len(statevector)))
    qc = QuantumCircuit(qubits)
    qc.initialize(statevector, qc.qubits)
    img_path = "quantum_attention_circuit.png"
    circuit_drawer(qc, output="mpl", filename=img_path)
    Image.open(img_path).show()

def getUnitaryFromTk(psi):
    import numpy as np

    # Normalizza psi
    psi = psi / np.linalg.norm(psi)
    dimensione = len(psi)
    
    # Inizializza la base con psi
    base = [psi]
    

    # Costruisci le colonne ortogonali con Gram-Schmidt
    while len(base) < dimensione:
        

        vec = np.random.rand(dimensione) + 1j * np.random.rand(dimensione)
        

        # Ortogonalizza rispetto ai vettori già nella base
        for i, b in enumerate(base):
            coeff = np.vdot(b, vec)
            
            vec -= coeff * b

        # Normalizza il nuovo vettore
        norm = np.linalg.norm(vec)
        
        if norm < 1e-12:
            print("  Norma troppo piccola, scarto vettore e riprovo.")
            continue

        vec /= norm
        base.append(vec)
        

    # Colonne → matrice unitaria
    U = np.column_stack(base)
    


    return U

def get_params(_num_qubits, _num_layer):
        # Per ora uso n_qubit e n_layer fissi
        num_qubits = _num_qubits
        x = get_param_resolver(num_qubits, _num_layer)
        params = getParamsShape(x, num_qubits, _num_layer)
        return params

def get_param_resolver(num_qubits, num_layers):
    num_angles = 12 * num_qubits * num_layers
    angs = np.pi * (2 * np.random.rand(num_angles) - 1)
    params = ParameterVector('θ', num_angles)
    param_dict = dict(zip(params, angs))
    return param_dict

def getParamsShape(param_list, num_qubits, num_layer):
        # Verifica il numero totale di elementi
        # assert len(param_list) % 6 == 0, "invalid number of parameters"
    param_values = np.array(list(param_list.values()))  # ho tolto .values per una migliore visualizzazione
    x = param_values.reshape(num_layer, 2, num_qubits // 2, 12)
    x_reshaped = x.reshape(num_layer, 2, num_qubits // 2, 4, 3)
    return x_reshaped

def makeInitialCircuit(psi,paramsV,paramsK,numLayer):
    
    n_qubits = int(np.log2(len(psi)))
    half = n_qubits // 2
    ansatz_v = ansatzBuilder(half ,paramsV,numLayer)
    ansatz_k = ansatzBuilder(half ,paramsK,numLayer)
    
    #ansatz_q = ansatzBuilder(half ,paramsQ)
    U = getUnitaryFromTk(psi.data)
    

    gate_U = UnitaryGate(U, #label="Uψ1"
                         )
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.append(gate_U, list(range(n_qubits)))
    
    # Applica le matrici unitarie agli stessi gruppi
    qc.compose(ansatz_v.get_unitary("V"), list(range(half)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(half, n_qubits)), inplace=True)
   
    return qc

def getCircuitUXDaggerFromTk(t_k):

    t_k = np.array(t_k, dtype=complex)
    t_k = t_k / np.linalg.norm(t_k)
    dim = len(t_k)
    n = int(np.log2(dim))

    U = getUnitaryFromTk(t_k)
    U_dagger = U.conj().T
    gate = UnitaryGate(U_dagger, 
                       #label="U†_x"
                       )
    qc = QuantumCircuit(n, name="U†_x")
    qc.append(gate, range(n))

    return qc

def getLossFromPsi(psi, V, K, numLayer):
    #print("Psi:", psi)
    n_qubits = int(np.log2(len(psi)))
    half = n_qubits // 2
    lunghezza_tk = int(np.sqrt(len(psi.data)))  # perché t_k ⊗ t_k ha len² ampiezze
    psi_matrix = psi.data.reshape((lunghezza_tk, lunghezza_tk))
    n_token = psi_matrix.shape[1]
    nShots = 1024*5
    total_loss = 0.0
    for i in range(n_token - 1):  # ATTENZIONE: fino a n_token - 1
        
        t_k = psi_matrix[:, i]
        norm = np.linalg.norm(t_k)

        if np.isnan(norm) or norm == 0:
            #print(f"⚠️ Salto t_k[{i}] per norma nulla o NaN")
            continue
          
        
        try:
            U_dagger = getCircuitUXDaggerFromTk(t_k)
        except Exception as e:
            print(f"Errore nella costruzione di U† per t_k[{i}]: {e}")
            continue
        
        circuit = makeInitialCircuit(psi, V, K, numLayer)
        
        circuit.append(U_dagger.to_instruction(#label=f"U†x_{i}_B"
            ), list(range(half, n_qubits)))
        circuit.append(U_dagger.to_instruction(#label=f"U†x_{i}_A"
            ), list(range(half)))
        circuit.measure(list(range(n_qubits)), list(range(n_qubits)))  # misura solo A

        showCircuit(circuit)
        
        sim = Aer.get_backend("aer_simulator")
        result = sim.run(transpile(circuit, sim), shots = nShots).result()
        counts = result.get_counts()
        target = "0" * n_qubits
        num_all_zero = counts.get(target, 0)
        
        p_i = num_all_zero / nShots
        
        if p_i > 0:
            total_loss += -np.log(p_i)

    
    return total_loss


def getLossFromQuantumFrasi(V, K, numLayer, states):
    loss_totale = 0.0
    num_validi = 0
    print("frasi:", states)
    for i in range(1, len(states)):  
        psi = None  # inizializza come None

        for j in range(i):
            t = states[j]
            #print(f"Stato t{j}:", np.round(t, 5))
            t = t / np.linalg.norm(t)  # normalizza lo stato singolo
            kron = np.kron(t, t)       # fai il prodotto tensoriale t ⊗ t
            psi = kron if psi is None else psi + kron

        psi = psi / np.linalg.norm(psi)  # normalizza la somma finale
        #print("ψ somma tensoriali:", np.round(psi, 5))
        #print("Norma finale:", np.linalg.norm(psi))

        loss = getLossFromPsi(Statevector(psi), V, K, numLayer)
        loss_totale += loss
        num_validi += 1
    #print("FINISCO PERCHEè SONO A POSTO")
    return loss_totale / num_validi if num_validi > 0 else 0.0


def showCircuit(qc):
    img_path = "quantum_attention_circuit.png"
    circuit_drawer(qc, output="mpl", filename=img_path)
    Image.open(img_path).show()

 
def salva_parametri(parametri, nome_file="parametri_migliori.npz"):
    np.savez(nome_file, **parametri)

def carica_parametri(numLayer, n_qubit, nome_file = "parametri_migliori.npz"):
    # Carica il dizionario dei parametri salvati
    dati = np.load(nome_file)
    
    # Ricostruisce array piatto a partire dai parametri salvati come "parametro_i"
    lista_parametri = [dati[f"parametro_{i}"] for i in range(len(dati))]
    parametri_piatto = np.array(lista_parametri)

    # Calcola la forma target
    param_shape = get_params(n_qubit//2, numLayer).shape
    n_params = np.prod(param_shape)

    # Verifica di sicurezza
    assert len(parametri_piatto) >= 2 * n_params, \
        f"Parametri insufficienti: attesi almeno {2 * n_params}, trovati {len(parametri_piatto)}"

    # Ricostruzione dei tensori V e K
    V = parametri_piatto[:n_params].reshape(param_shape)
    K = parametri_piatto[n_params:2*n_params].reshape(param_shape)
    return {"V": V, "K": K}


def plot_loss_all(losses, best_losses, worst_losses, times = 0, nqubit = 16,nome_base="loss_plot"):
    # Primo grafico: rispetto alle iterazioni
    plt.figure()
    plt.plot(losses, marker="o", label="Loss")
    plt.plot(best_losses, marker="s", linestyle="--", label="Best Loss")
    plt.plot(worst_losses, marker="x", linestyle=":", label="Worst Loss")
    plt.title("Loss rispetto alle Iterazioni")
    plt.xlabel("Iterazione")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{nome_base}_qubit_e_iterazioni.png")
    plt.close()

    # Secondo grafico: rispetto al tempo
    """
    plt.figure()
    plt.plot(times, losses, marker="o", label="Loss")
    plt.plot(times, best_losses, marker="s", linestyle="--", label="Best Loss")
    plt.plot(times, worst_losses, marker="x", linestyle=":", label="Worst Loss")
    plt.title("Loss rispetto al Tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{nome_base}_{nqubit}qubit_e_tempo.png")
    plt.close()"""

def getLossFromFrasi(numFrasi, V, K, numLayer, enc):
    loss = 0.0
    
    
    for i in range(numFrasi):
        lossFrase = 0.0
        
        lenFrase = len(enc.getFrasi()[i])
        
        for e in range(1, lenFrase):
            
            psi = Statevector(enc.psi_locale(i,e))
            loss = getLossFromPsi(psi, V, K, numLayer)
            lossFrase += loss
        
        lossFrase /= (lenFrase-1)
        loss += lossFrase
    
    
    return loss

"""
def ottimizzazione(numFrasi, ore_max, numeroIterazioni, numLayer, enc):
    import json
    import time
    import numpy as np
    from datetime import datetime
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    metodi = ['COBYLA', 'Powell', 'SLSQP']
    print("\nInizio ottimizzazione dei parametri ansatz...\n")
    
    best_loss = float("inf")
    worst_loss = 0
    n_qubit = int(np.log2(len(Statevector(enc.psi_locale(0)).data))) // 2
    best_params = None
    param_shape = get_params(n_qubit, numLayer).shape
    n_params = np.prod(param_shape)
    stop_early = [False]
    losses = []
    worst_losses = []
    best_losses = []
    times = []
    inizio = time.time()
    iterazione = [0]
    timeout_secondi = ore_max * 3600
    ultima_loss = [None]
    ultimo_elapsed = [None]
    ultimi_params = [None, None]  # V, K
    for j in range(20):
        best_params = None
        lossesTemp = []
        for i in range(numeroIterazioni):
            if best_params is None:
                params_init = np.concatenate([
                    get_params(n_qubit, numLayer).flatten(),
                    get_params(n_qubit, numLayer).flatten(),
                ])
            else:
                params_init = np.concatenate([
                    best_params["V"].flatten(),
                    best_params["K"].flatten(),
                ])

            def salva_backup():
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                parametri = carica_parametri(numLayer, n_qubit)
                backup = {
                    "timestamp": timestamp,
                    "best_loss": best_loss,
                    "params": converti(parametri),
                }
                with open("quantum_backup.json", "w") as f:
                    json.dump(backup, f, indent=2)

            def loss_totale(params_tutti):
                nonlocal lossesTemp
                iterazione[0] += 1
                elapsed = time.time() - inizio
                ultimo_elapsed[0] = elapsed

                if elapsed > timeout_secondi:
                    salva_backup()
                    raise TimeoutError("Tempo massimo di ottimizzazione raggiunto.")

                if iterazione[0] % 100 == 0:
                    print(f"Iterazione {iterazione[0]}" + " a ", datetime.now().strftime("%H:%M:%S"))
                    salva_backup()

                if int(elapsed) % 1800 < 2:
                    mins = int(elapsed // 60)
                    print(f" Tempo trascorso: {mins} minuti")
                    salva_backup()

                pV = params_tutti[:n_params].reshape(param_shape)
                pK = params_tutti[n_params:2 * n_params].reshape(param_shape)
                ultimi_params[0] = pV
                ultimi_params[1] = pK

                loss = getLossFromFrasi(numFrasi, pV, pK, numLayer, enc)
                lossesTemp.append(loss)
                ultima_loss[0] = loss
                return loss

            try:
                if iterazione[0] < 100:
                    metodo = metodi[0]
                elif iterazione[0] < 1750:
                    print("Metodo 1")
                    metodo = metodi[1]
                else:
                    print("Metodo 2")
                    metodo = metodi[2]

                minimize(
                    loss_totale,
                    params_init,
                    method=metodo,
                    options={'maxiter': 250, 'disp': False}
                )
                lossTemp = sum(lossesTemp) 
            lossTemp = lossTemp/numeroIterazioni
            losses.append(lossTemp)
            #Voglio    

                if lossTemp > worst_loss:
                    worst_loss = lossTemp
                if lossTemp < best_loss:
                    best_loss = lossTemp
                    print(f" Nuova loss migliore trovata: {best_loss:.6f}" + " a ", datetime.now().strftime("%H:%M:%S"))
                    best_params = {
                        "V": ultimi_params[0].copy(),
                        "K": ultimi_params[1].copy(),
                    }

                losses.append(lossTemp)
                worst_losses.append(worst_loss)
                best_losses.append(best_loss)
                plot_loss_all(losses, best_losses, worst_losses)

                if stop_early[0]:
                    print("\n✅ Ottimizzazione interrotta manualmente per low loss.")
                    break

            except TimeoutError:
                print("\n⏹️ Interrotto: tempo massimo raggiunto.")
                break

        print("\n Ottimizzazione terminata.")
        print(" Loss migliore:", best_loss)

        plot_loss_all(losses, best_losses, worst_losses, times)

    # Salva i parametri finali

    return """

def ottimizzazioneQuantum(ore_max, numeroIterazioni, numLayer, quantumStates):
    import json
    import time
    import numpy as np
    from datetime import datetime
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    print("Quantum states:", quantumStates)
    metodi = ['COBYLA', 'Powell', 'SLSQP']
    print("\nInizio ottimizzazione dei parametri ansatz...\n")

    best_loss = float("inf")
    
    
    n_qubit = 4
    best_params = None
    param_shape = get_params(n_qubit, numLayer).shape
    n_params = np.prod(param_shape)
    inizio = time.time()
    numMaxiter = 250
    iterazione = [0]
    timeout_secondi = ore_max * 3600
    ultima_loss = [None]
    ultimo_elapsed = [None]
    ultimi_params = [None, None]  # V, K
    lossTotaliSalvate = []
    numEsperimenti = 1

    for j in range(numEsperimenti):
        print(f"\nInizio ottimizzazione per iterazione {j}...")
        best_params = None
        lossesTemp = []

        for i in range(numeroIterazioni):

            if best_params is None:
                print(f"Inizializzazione dei parametri per iterazione {j}...")
                params_init = np.concatenate([
                    get_params(n_qubit, numLayer).flatten(),
                    get_params(n_qubit, numLayer).flatten(),
                ])
            else:
                print(f"Inizializzazione dei parametri per iterazione {j} con i migliori trovati finora...")
                parametri_piatto = np.array(best_params).flatten()

                # Calcola la forma target
                param_shape = get_params(n_qubit, numLayer).shape
                n_params = np.prod(param_shape)
                V = parametri_piatto[:n_params].reshape(param_shape)
                K = parametri_piatto[n_params:2*n_params].reshape(param_shape)
                params_init = np.concatenate([
                    V.flatten(),
                    K.flatten(),
                ])

            def salva_backup():
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                parametri = carica_parametri(numLayer, n_qubit)
                backup = {
                    "timestamp": timestamp,
                    "best_loss": best_loss,
                    "params": converti(parametri),
                }
                with open("quantum_backup.json", "w") as f:
                    json.dump(backup, f, indent=2)

            def loss_totale(params_tutti):
                nonlocal lossesTemp
                iterazione[0] += 1
                elapsed = time.time() - inizio
                ultimo_elapsed[0] = elapsed

                if elapsed > timeout_secondi:
                    
                    raise TimeoutError("Tempo massimo di ottimizzazione raggiunto.")

                if iterazione[0] % 100 == 0:
                    print(f"Iterazione {iterazione[0]}" + " a ", datetime.now().strftime("%H:%M:%S"))
                    

                if int(elapsed) % 1800 < 2:
                    mins = int(elapsed // 60)
                    print(f" Tempo trascorso: {mins} minuti")
                    
                
                pV = params_tutti[:n_params].reshape(param_shape)
                pK = params_tutti[n_params:2 * n_params].reshape(param_shape)
                ultimi_params[0] = pV
                ultimi_params[1] = pK

                loss = getLossFromQuantumFrasiMultiple(pV, pK, numLayer, quantumStates)
                
                lossesTemp.append(loss)
                
                return loss

            try:
                if numeroIterazioni < 500:
                    metodo = metodi[0]
                else:
                    metodo = metodi[1]

                a = minimize(
                    loss_totale,
                    params_init,
                    method=metodo,
                    options={'maxiter': numMaxiter, 'disp': False}
                )
                best_params = a.x


            except TimeoutError:
                print("\n⏹️ Interrotto: tempo massimo raggiunto.")
                break
        lossTotaliSalvate.append(lossesTemp)
    media_per_iterazione = [sum(x)/len(x) for x in zip(*lossTotaliSalvate)]
    lossBest = [min(x) for x in zip(*lossTotaliSalvate)]
    lossWorst = [max(x) for x in zip(*lossTotaliSalvate)]
    salva_grafico_loss(media_per_iterazione, lossBest, lossWorst, numLayer)
    salva_valori_loss_su_file(media_per_iterazione, lossBest, lossWorst, "loss_risultati.txt")


    print(lossTotaliSalvate)
    return best_params


def ottimizzazioneClassic(ore_max, numeroIterazioni, numLayer, numFrasi, enc):
    import json
    import time
    import numpy as np
    from datetime import datetime
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    metodi = ['COBYLA', 'Powell', 'SLSQP']
    print("\nInizio ottimizzazione dei parametri ansatz...\n")

    best_loss = float("inf")
    
    lenParola = len(enc.getFrasi()[0])
    n_qubit = int(np.log2(len(Statevector(enc.psi_locale(0,lenParola-1)).data))) // 2
    print("N_qubit:", n_qubit)
    best_params = None
    param_shape = get_params(n_qubit, numLayer).shape
    n_params = np.prod(param_shape)
    inizio = time.time()
    numMaxiter = 250
    iterazione = [0]
    timeout_secondi = ore_max * 3600
    ultima_loss = [None]
    ultimo_elapsed = [None]
    ultimi_params = [None, None]  # V, K
    lossTotaliSalvate = []
    numEsperimenti = 1

    for j in range(numEsperimenti):
        print(f"\nInizio ottimizzazione per iterazione {j}...")
        best_params = None
        lossesTemp = []

        for i in range(numeroIterazioni):

            if best_params is None:
                print(f"Inizializzazione dei parametri per iterazione {j}...")
                params_init = np.concatenate([
                    get_params(n_qubit, numLayer).flatten(),
                    get_params(n_qubit, numLayer).flatten(),
                ])
            else:
                print(f"Inizializzazione dei parametri per iterazione {j} con i migliori trovati finora...")
                parametri_piatto = np.array(best_params).flatten()

                # Calcola la forma target
                param_shape = get_params(n_qubit, numLayer).shape
                n_params = np.prod(param_shape)
                V = parametri_piatto[:n_params].reshape(param_shape)
                K = parametri_piatto[n_params:2*n_params].reshape(param_shape)
                params_init = np.concatenate([
                    V.flatten(),
                    K.flatten(),
                ])

            def salva_backup():
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                parametri = carica_parametri(numLayer, n_qubit)
                backup = {
                    "timestamp": timestamp,
                    "best_loss": best_loss,
                    "params": converti(parametri),
                }
                with open("quantum_backup.json", "w") as f:
                    json.dump(backup, f, indent=2)

            def loss_totale(params_tutti):
                nonlocal lossesTemp
                iterazione[0] += 1
                elapsed = time.time() - inizio
                ultimo_elapsed[0] = elapsed

                if elapsed > timeout_secondi:
                    
                    raise TimeoutError("Tempo massimo di ottimizzazione raggiunto.")

                if iterazione[0] % 100 == 0:
                    print(f"Iterazione {iterazione[0]}" + " a ", datetime.now().strftime("%H:%M:%S"))
                    

                if int(elapsed) % 1800 < 2:
                    mins = int(elapsed // 60)
                    print(f" Tempo trascorso: {mins} minuti")
                    
                
                pV = params_tutti[:n_params].reshape(param_shape)
                pK = params_tutti[n_params:2 * n_params].reshape(param_shape)
                ultimi_params[0] = pV
                ultimi_params[1] = pK

                loss = getLossFromFrasi(numFrasi, pV, pK, numLayer, enc)
                
                lossesTemp.append(loss)
                
                return loss

            try:
                metodo = metodi[0]

                a = minimize(
                    loss_totale,
                    params_init,
                    method=metodo,
                    options={'maxiter': numMaxiter, 'disp': False}
                )
                best_params = a.x


            except TimeoutError:
                print("\n⏹️ Interrotto: tempo massimo raggiunto.")
                break
        lossTotaliSalvate.append(lossesTemp)
    media_per_iterazione = [sum(x)/len(x) for x in zip(*lossTotaliSalvate)]
    lossBest = [min(x) for x in zip(*lossTotaliSalvate)]
    lossWorst = [max(x) for x in zip(*lossTotaliSalvate)]
    salva_grafico_loss(media_per_iterazione, lossBest, lossWorst, numLayer)
    salva_valori_loss_su_file(media_per_iterazione, lossBest, lossWorst, "loss_risultati.txt")



    return best_params

import matplotlib.pyplot as plt

def salva_valori_loss_su_file(media, best, worst, filename="valori_loss.txt"):
    """
    Salva i valori di loss media, migliore e peggiore per iterazione in un file.

    Args:
        media (list): Lista delle medie per iterazione.
        best (list): Lista dei minimi per iterazione.
        worst (list): Lista dei massimi per iterazione.
        filename (str): Nome del file da creare.
    """
    with open(filename, "w") as f:
        f.write("Iterazione\tLoss Media\tLoss Migliore\tLoss Peggiore\n")
        for i in range(len(media)):
            riga = f"{i}\t{media[i]:.6f}\t{best[i]:.6f}\t{worst[i]:.6f}\n"
            f.write(riga)

    print(f"Valori salvati in: {filename}")


def salva_grafico_loss(media_per_iterazione, lossBest, lossWorst, numLayer,filename="grafico_loss.png"):
    """
    Salva un grafico della loss media, migliore e peggiore in un file PNG.

    Args:
        media_per_iterazione (list): Lista delle medie delle loss per iterazione.
        lossBest (list): Lista dei minimi per iterazione.
        lossWorst (list): Lista dei massimi per iterazione.
        filename (str): Nome del file in cui salvare il grafico.
    """
    filename = f"grafico_loss_{numLayer}layer.png"
    x = range(len(media_per_iterazione))

    plt.figure(figsize=(10, 5))
    plt.plot(x, media_per_iterazione, label='Loss Media', linewidth=2)
    plt.plot(x, lossBest, label='Loss Migliore (Best)', linestyle='--', color='green')
    plt.plot(x, lossWorst, label='Loss Peggiore (Worst)', linestyle='--', color='red')

    plt.xlabel('Iterazione')
    plt.ylabel('Loss')
    plt.title('Andamento delle Loss per Iterazione')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()  # così non mostra nulla a video

    print(f"Grafico salvato in: {filename}")


def converti(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, dict):
        return {k: converti(v) for k, v in o.items()}
    else:
        return o

def get_frasi_dataset(split="train", max_frasi=100):
    dataset_dict = load_dataset("ptb_text_only", trust_remote_code=True)
    if split not in dataset_dict:
        raise ValueError(f"Split '{split}' non disponibile! Gli split validi sono: {list(dataset_dict.keys())}")
    
    dataset = dataset_dict[split]
    frasi = []
    for entry in dataset:
        frase = entry["sentence"]
        if len(frase.split()) >= 4:
            frasi.append(frase)
        if len(frasi) >= max_frasi:
            break
    return frasi


def getLossFromQuantumFrasiMultiple(V, K, numLayer, states):
    loss_totale = 0.0
    num_frasi = len(states)
    

    for frase in states:
        loss_frase = 0.0
        lunghezza = len(frase)
        
        for indexParola in range(1, lunghezza):
            psi = None
                               
            for j in range(indexParola):  
                t = np.array(frase[j], dtype=np.complex128)
                t = t / np.linalg.norm(t)
                kron = np.kron(t, t)
                psi = kron if psi is None else psi + kron

            psi = psi / np.linalg.norm(psi)
            stato = Statevector(psi)
            loss = getLossFromPsi(stato, V, K, numLayer)
            loss_frase += loss

        loss_frase /= (lunghezza-1) 
        loss_totale += loss_frase

    return loss_totale

def clean_statevectors(states, soglia=1e-6):
    cleaned_all = []
    for state_list in states:  # ogni riga della lista
        cleaned_list = []
        for s in state_list:
            vec = s.data.copy()
            vec[np.abs(vec) < soglia] = 0.0
            norm = np.linalg.norm(vec)
            if norm == 0:
                raise ValueError("Statevector annullato completamente.")
            vec = vec / norm
            cleaned_list.append(Statevector(vec))
        cleaned_all.append(cleaned_list)
    return cleaned_all

if __name__ == "__main__":
    numIterazioni = 2 #multiplo di 250
    numLayers = 3
    ore_max = 100 
    
    frasi = ["We love pizza",
             "Timeflies fast",
             "Robots need power",
             "Birds can sing",
             "Dogs are friendly",
             "I trust you",
             "Machines learn quickly"
        
        
    ]
    
    #train_frasi = get_frasi_dataset(split="train", max_frasi=numFrasi)
    numFrasi = 3 #SANDRO QUI metti 6
    #enc = Encoding(frasi, embedding_dim=16)
    print("Inizio ottimizzazione")
    
    #ottimizzazioneClassic(ore_max, numIterazioni, numLayers,numFrasi, enc)
    all_states = []
    for _ in range(numFrasi):
        seed = random.randint(0, 10000)
        all_states.append(QuantumEvolver(n_qubit=4, max_time=6, seed = seed).get_states())
    ottimizzazioneQuantum(ore_max, numIterazioni, numLayers,all_states)
   
    
    
