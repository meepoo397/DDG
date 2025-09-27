import numpy as np
def const_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    return guidance

def linear_increase_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    guidance = guidance - 1
    return 2*(i/num_steps)*guidance + 1

def linear_decrease_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    guidance = guidance - 1
    return 2*(1-i/num_steps)*guidance + 1

def late_activate_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    guidance = guidance - 1
    return 2*guidance+1 if i>num_steps/2 else 1

def mid_activate_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    low = schedule_kwargs.get('w_interval_low_steps',num_steps/2)
    high = schedule_kwargs.get('w_interval_high_steps',num_steps*4/5)
    return guidance if i>low and i<=high else 1
def interval_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    # Pass hyperparameter by arguments of edm2implfrom generate_images_for_test.py
    sigma_low = schedule_kwargs.get('w_interval_low_sigma',0.28)
    sigma_high = schedule_kwargs.get('w_interval_high_sigma',2.9)
    return guidance if t>sigma_low and t<sigma_high else 1
def trapezoid_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    low = schedule_kwargs.get('w_interval_low_steps',num_steps*5/10)
    low_peak = schedule_kwargs.get('w_interval_low_peak_steps',num_steps*6/10)
    high_peak = schedule_kwargs.get('w_interval_high_peak_steps',num_steps*7/10)
    high = schedule_kwargs.get('w_interval_high_steps',num_steps*8/10)
    # w=1 (low) w linear increase (low_peak) w=guidance (high_peak) w linear decrease (high) w=1
    assert 0<=low and low<=low_peak and low_peak <= high_peak and high_peak <=high
    if i<=low or i>high:
        return 1
    elif i>low and i<=low_peak:
        return (guidance-1)*(i-low)/(low_peak-low)+1
    elif i>low_peak and i<=high_peak:
        return guidance
    else:# i>high_peak and i<=high:
        return (guidance-1)*(high-i)/(high-high_peak)+1
def skewed_sinusoidal_schedule(i,num_steps,guidance,t=0,**schedule_kwargs):
    skewed = schedule_kwargs.get('w_skewed',0.38)
    return (guidance-1)*np.sin(((num_steps - i -1) / num_steps) ** skewed * np.pi)+1
def polygon_schedule(i,num_steps,guidance,t=0,**schedule_kwargs):
    low = schedule_kwargs.get('w_interval_low_steps',num_steps*5/10)
    low_peak = schedule_kwargs.get('w_interval_low_peak_steps',num_steps*6/10)
    high_peak = schedule_kwargs.get('w_interval_high_peak_steps',num_steps*7/10)
    high_middle = schedule_kwargs.get('w_interval_high_middle_steps',num_steps*8/10)
    high = schedule_kwargs.get('w_interval_high_steps',num_steps*8/10)
    # w=1 (low) w linear increase (low_peak) w=guidance (high_peak) w linear decrease (high) w=1
    assert 0<=low and low<=low_peak and low_peak <= high_peak and high_peak <=high
    if i<=low or i>high:
        return 1
    elif i>low and i<=low_peak:
        return (guidance-1)*(i-low)/(low_peak-low)+1
    elif i>low_peak and i<=high_peak:
        return guidance
    elif i>high_peak and i<=high_middle:
        return (guidance-1)/2*(high_middle-i)/(high_middle-high_peak)+1+(guidance-1)/2
    else:# i>high_peak and i<=high:
        return (guidance-1)/2*(high-i)/(high-high_middle)+1
def restricted_linear_increase_scheduler(i,num_steps,guidance,t=0,**schedule_kwargs):
    guidance = guidance - 1
    low = schedule_kwargs.get('w_interval_low_steps',num_steps/2)
    high = schedule_kwargs.get('w_interval_high_steps',num_steps*4/5)
    return 2*(i/num_steps)*guidance + 1 if i>low and i<=high else 1
def restricted_skewed_sinusoidal_schedule(i,num_steps,guidance,t=0,**schedule_kwargs):
    low = schedule_kwargs.get('w_interval_low_steps',5)
    high = schedule_kwargs.get('w_interval_high_steps',30)
    skewed = schedule_kwargs.get('w_skewed',0.39)
    return (guidance-1)*np.sin(((num_steps - i - 1) / num_steps) ** skewed * np.pi) + 1 if i>low and i<=high else 1

guidance_scheduler_config = {
    'linear_increase_scheduler': linear_increase_scheduler,
    'linear_decrease_scheduler': linear_decrease_scheduler,
    'late_activate_scheduler': late_activate_scheduler,
    'mid_activate_scheduler': mid_activate_scheduler,
    'const_scheduler': const_scheduler,
    'interval_scheduler': interval_scheduler,
    'trapezoid_scheduler':trapezoid_scheduler,
    'skewed_sinusoidal_schedule':skewed_sinusoidal_schedule,
    'polygon_schedule':polygon_schedule,
    'restricted_linear_increase_scheduler':restricted_linear_increase_scheduler,
    'restricted_skewed_sinusoidal_schedule':restricted_skewed_sinusoidal_schedule
}