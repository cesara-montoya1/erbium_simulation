"""
erbium_plotting.py
==================
Funciones de graficación para la simulación de iones de Erbio - VERSIÓN ACTUALIZADA
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_static_plots(simulator, t, N0, N1, N2, title_suffix="", laser_off_time=None):
    """
    Crea gráficos estáticos de la simulación con tasa de emisión MEJORADA
    CAMBIO PRINCIPAL: Reemplaza gráfico poco útil de N2 con análisis detallado de emisión
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Calcular tasa de emisión
    emission_rate = simulator.calculate_emission_rate(N1)
    
    # Gráfico 1: Dinámicas de población (solo niveles 0 y 1) - SIN CAMBIOS
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t * 1000, N0, 'b-', linewidth=2, label='Nivel 0 (fundamental)')
    ax1.plot(t * 1000, N1, 'r-', linewidth=2, label='Nivel 1 (metaestable, 1530 nm)')
    
    if laser_off_time is not None:
        ax1.axvline(x=laser_off_time * 1000, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Láser apagado')
    
    ax1.set_xlabel('Tiempo (ms)')
    ax1.set_ylabel('Número de iones')
    ax1.set_title(f'Dinámica de Poblaciones Er³⁺{title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico 2: Tasa de emisión óptica - MEJORADO
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t * 1000, emission_rate, 'orange', linewidth=3, label='Tasa de emisión')
    ax2.fill_between(t * 1000, emission_rate, alpha=0.3, color='orange')
    
    if laser_off_time is not None:
        ax2.axvline(x=laser_off_time * 1000, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Láser apagado')
    
    ax2.set_xlabel('Tiempo (ms)')
    ax2.set_ylabel('Tasa de emisión (fotones/s)')
    ax2.set_title('Tasa de Emisión Óptica (N₁/τ₁₀)')  # TÍTULO ACTUALIZADO
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Formato científico para el eje Y
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # MEJORA: Añadir información de pico máximo
    max_emission_idx = np.argmax(emission_rate)
    max_emission_rate = emission_rate[max_emission_idx]
    max_time = t[max_emission_idx] * 1000
    
    ax2.plot(max_time, max_emission_rate, 'ro', markersize=8, zorder=5)
    ax2.annotate(f'Máx: {max_emission_rate:.1e} fotones/s\n@ {max_time:.1f} ms',
                xy=(max_time, max_emission_rate),
                xytext=(max_time + (t[-1]*1000*0.1), max_emission_rate * 0.8),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=9)
    
    # Gráfico 3: Diagrama de niveles energéticos - SIN CAMBIOS
    ax3 = plt.subplot(2, 3, 3)
    draw_energy_levels(ax3, simulator, N0[-1], N1[-1], N2[-1])
    
    # Gráfico 4: Comparación N1 vs Tasa de emisión - MEJORADO
    ax4 = plt.subplot(2, 3, 4)
    
    # Normalizar para comparación visual
    N1_norm = N1 / np.max(N1)
    emission_norm = emission_rate / np.max(emission_rate)
    
    ax4.plot(t * 1000, N1_norm, 'r-', linewidth=2, label='N₁ (normalizado)', alpha=0.8)
    ax4.plot(t * 1000, emission_norm, 'orange', linewidth=2, linestyle='--', 
            label='Tasa emisión (normalizada)', alpha=0.9)
    
    # MEJORA: Rellenar áreas para mejor visualización
    ax4.fill_between(t * 1000, N1_norm, alpha=0.2, color='red')
    ax4.fill_between(t * 1000, emission_norm, alpha=0.2, color='orange')
    
    if laser_off_time is not None:
        ax4.axvline(x=laser_off_time * 1000, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax4.set_xlabel('Tiempo (ms)')
    ax4.set_ylabel('Valor normalizado')
    ax4.set_title('Correlación: N₁ ↔ Tasa de Emisión')  # TÍTULO MEJORADO
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    # Gráfico 5: NUEVO - Análisis de decaimiento (reemplaza el log anterior menos útil)
    ax5 = plt.subplot(2, 3, 5)
    
    if laser_off_time is not None:
        # Análisis específico del decaimiento
        off_idx = np.where(t >= laser_off_time)[0]
        if len(off_idx) > 0:
            t_decay = t[off_idx[0]:] * 1000
            emission_decay = emission_rate[off_idx[0]:]
            
            ax5.semilogy(t_decay, emission_decay, 'red', linewidth=3, 
                        label='Decaimiento exponencial')
            
            # Ajuste exponencial para mostrar τ efectivo
            if len(emission_decay) > 10:
                # Ajuste exponencial simple
                t_fit = t_decay - t_decay[0]
                valid_idx = emission_decay > np.max(emission_decay) * 0.01
                if np.sum(valid_idx) > 5:
                    log_emission = np.log(emission_decay[valid_idx])
                    tau_eff = -1 / np.polyfit(t_fit[valid_idx]/1000, log_emission, 1)[0]
                    
                    # Curva ajustada
                    fit_curve = emission_decay[0] * np.exp(-(t_fit/1000) / tau_eff)
                    ax5.semilogy(t_decay, fit_curve, 'orange', linewidth=2, 
                                linestyle='--', label=f'Ajuste: τ≈{tau_eff*1000:.1f} ms')
    else:
        # Si no hay apagado, mostrar toda la curva en log
        ax5.semilogy(t * 1000, emission_rate, 'orange', linewidth=2, 
                    label='Tasa de emisión (log)')
    
    ax5.semilogy(t * 1000, N1, 'g--', linewidth=1.5, alpha=0.6, 
                label='N₁ (referencia)')
    
    if laser_off_time is not None:
        ax5.axvline(x=laser_off_time * 1000, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Láser apagado')
    
    ax5.set_xlabel('Tiempo (ms)')
    ax5.set_ylabel('Intensidad (escala log)')
    ax5.set_title('Análisis de Decaimiento Exponencial')  # TÍTULO MEJORADO
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Gráfico 6: Espectro de emisión - SIN CAMBIOS (ya está bien)
    ax6 = plt.subplot(2, 3, 6)
    wavelengths, spectrum = simulator.generate_emission_spectrum(N1[-1])
    ax6.plot(wavelengths, spectrum, 'r-', linewidth=2)
    ax6.fill_between(wavelengths, spectrum, alpha=0.3, color='red')
    ax6.set_xlabel('Longitud de onda (nm)')
    ax6.set_ylabel('Intensidad de emisión (u.a.)')
    ax6.set_title('Espectro de Emisión Er³⁺')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(1450, 1610)
    
    # Marcar pico principal
    peak_idx = np.argmax(spectrum)
    ax6.annotate(f'Pico: {wavelengths[peak_idx]:.0f} nm', 
                xy=(wavelengths[peak_idx], spectrum[peak_idx]),
                xytext=(wavelengths[peak_idx] + 20, spectrum[peak_idx] * 0.8),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    plt.tight_layout()
    
    # MEJORA: Imprimir estadísticas de la emisión
    print("\n" + "="*50)
    print("ESTADÍSTICAS DE EMISIÓN ÓPTICA")
    print("="*50)
    print(f"Tasa máxima: {max_emission_rate:.2e} fotones/s")
    print(f"Tiempo del máximo: {max_time:.2f} ms")
    print(f"τ₁₀ = {simulator.tau10*1000:.1f} ms")
    print(f"Factor 1/τ₁₀ = {1/simulator.tau10:.1f} s⁻¹")
    
    if laser_off_time is not None:
        off_idx = np.where(t >= laser_off_time)[0][0]
        initial_decay = emission_rate[off_idx]
        final_decay = emission_rate[-1]
        print(f"\nDecaimiento (láser apagado en {laser_off_time*1000:.1f} ms):")
        print(f"Emisión inicial: {initial_decay:.2e} fotones/s")
        print(f"Emisión final: {final_decay:.2e} fotones/s")
        print(f"Reducción: {initial_decay/final_decay if final_decay > 0 else np.inf:.1f}x")
    print("="*50)
    
    return fig, emission_rate

def draw_energy_levels(ax, simulator, N0, N1, N2):
    """
    Dibuja diagrama de niveles energéticos con poblaciones - SIN CAMBIOS
    """
    # Posiciones de los niveles
    levels = [0, 1, 2]
    energies = [0, 1, 1.8]  # Energías relativas
    
    # Dibujar niveles
    for i, (level, energy) in enumerate(zip(levels, energies)):
        populations = [N0, N1, N2]
        width = populations[i] / simulator.N0 * 0.4  # Ancho proporcional a la población
        
        # Línea del nivel energético
        ax.hlines(energy, 0.3, 0.7, colors='black', linewidth=2)
        
        # Rectángulo representando la población
        rect = patches.Rectangle((0.5 - width/2, energy - 0.05), width, 0.1, 
                               linewidth=1, edgecolor='black', 
                               facecolor=['blue', 'red', 'green'][i], alpha=0.6)
        ax.add_patch(rect)
        
        # Etiquetas
        ax.text(0.8, energy, f'Nivel {level}\nN={populations[i]:.1e}', 
               va='center', fontsize=9)
    
    # Flechas de transición
    # Bombeo (0→2)
    ax.annotate('', xy=(0.15, 1.8), xytext=(0.15, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(0.05, 0.9, '980 nm\n(bombeo)', rotation=90, ha='center', va='center', fontsize=8)
    
    # Relajación (2→1)
    ax.annotate('', xy=(0.25, 1), xytext=(0.25, 1.8),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='orange', linestyle='--'))
    ax.text(0.27, 1.4, 'relajación\nno radiativa', ha='left', va='center', fontsize=7)
    
    # Emisión (1→0) - Destacar esta transición
    ax.annotate('', xy=(0.35, 0), xytext=(0.35, 1),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(0.37, 0.5, '1530 nm\n(EMISIÓN)', ha='left', va='center', fontsize=8, 
            weight='bold', color='red')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 2.2)
    ax.set_ylabel('Energía (u.a.)')
    ax.set_title('Diagrama de Niveles\nEr³⁺')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3)