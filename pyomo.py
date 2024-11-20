# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:55:05 2024

@author: DELL
"""

from pyomo.environ import *

def create_pyomo_model(instance, model_integer=False):
    model = ConcreteModel()

    # Extract instance data
    n_acopios = instance['n_acopios']
    n_centros = instance['n_centros']
    n_plantas = instance['n_plantas']
    n_productores = instance['n_productores']
    n_envases = instance['n_envases']
    n_periodos = instance['n_periodos']
    inflation = instance['inflation']
    qc = instance['qc']
    qt = instance['qt']
    ql = instance['ql']
    qb = instance['qb']
    ta = instance['ta']
    tl = instance['tl']
    qa = instance['qa']
    wa = instance['wa']
    em = instance['em']
    el = instance['el']
    et = instance['et']
    en = instance['en']
    recup_increm = instance['recup_increm']
    ccv = instance['ccv']
    acv = instance['acv']
    lpl = instance['lpl']
    apl = instance['apl']
    arr_cv = instance['arr_cv']
    arr_pl = instance['arr_pl']
    ade_cv = instance['ade_cv']
    ade_pl = instance['ade_pl']
    dep = instance['dep']
    enr = instance['enr']
    tri = instance['tri']
    adem = instance['adem']
    recup = instance['recup']
    cinv = instance['cinv']
    pinv = instance['pinv']
    envn = instance['envn']
    cc = instance['cc']
    ca = instance['ca']
    cp = instance['cp']
    cl = instance['cl']
    da = instance['da']
    dl = instance['dl']
    dp = instance['dp']
    rv = instance['rv']
    rl = instance['rl']
    av = instance['av']
    al = instance['al']
    qd = instance['qd']
    pv = instance['pv']
    pt = instance['pt']
    b = instance['b']
    dem = instance['dem']
    de = instance['de']
    gi = instance['gi']
    ge = instance['ge']
    iv = instance['iv']
    il = instance['il']
    ci = instance['ci']
    cv = instance['cv']
    pe = instance['pe']
    a = instance['a']
    acopios = instance['acopios']
    centros = instance['centros']
    plantas = instance['plantas']
    productores = instance['productores']
    envases = instance['envases']
    periodos = instance['periodos']

    # Define sets
    model.Centros = Set(initialize=centros)
    model.Plantas = Set(initialize=plantas)
    model.Acopios = Set(initialize=acopios)
    model.Productores = Set(initialize=productores)
    model.Envases = Set(initialize=envases)
    model.Periodos = Set(initialize=periodos)

    # Define variables
    if model_integer:
        vtype = NonNegativeIntegers
    else:
        vtype = NonNegativeReals

    model.x = Var(model.Centros, model.Periodos, domain=Binary)
    model.y = Var(model.Centros, model.Periodos, domain=Binary)
    model.z = Var(model.Plantas, model.Periodos, domain=Binary)
    model.w = Var(model.Plantas, model.Periodos, domain=Binary)
    model.q = Var(model.Envases, model.Acopios, model.Centros, model.Periodos, domain=vtype)
    model.r = Var(model.Envases, model.Centros, model.Plantas, model.Periodos, domain=vtype)
    model.u = Var(model.Envases, model.Plantas, model.Productores, model.Periodos, domain=vtype)
    model.ic = Var(model.Envases, model.Centros, model.Periodos, domain=vtype)
    model.ip = Var(model.Envases, model.Plantas, model.Periodos, domain=vtype)

    
    # Objective Function
    # Define ingreso_retornable
    def ingreso_retornable_rule(m):
        return sum(
            m.u[p, k, l, t] * pv[p] * (1 + inflation) ** ((t - 1) // 12)
            for p in m.envases
            for k in m.plantas
            for l in m.productores
            for t in m.periodos
            if (p, k, l, t) in m.combinations_u
        )
    model.ingreso_retornable = Expression(rule=ingreso_retornable_rule)
    
    # Define ingreso_triturado
    def ingreso_triturado_rule(m):
        return (
            sum(
                m.r[p, j, k, t] * (1 - ta) / ta * pt[p] * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for j in m.centros
                for k in m.plantas
                for t in m.periodos
            )
            + sum(
                m.u[p, k, l, t] * (1 - tl) / tl * pt[p] * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for k in m.plantas
                for l in m.productores
                for t in m.periodos
                if (p, k, l, t) in m.combinations_u
            )
        )
    model.ingreso_triturado = Expression(rule=ingreso_triturado_rule)
    
    # Define egreso_adecuar
    def egreso_adecuar_rule(m):
        return (
            sum(m.y[j, t] * av[j, t] for j in m.centros for t in m.periodos)
            + sum(m.w[k, t] * al[k, t] for k in m.plantas for t in m.periodos)
        )
    model.egreso_adecuar = Expression(rule=egreso_adecuar_rule)
    
    # Define egreso_uso
    def egreso_uso_rule(m):
        return (
            sum(m.x[j, t] * rv[j, t] for j in m.centros for t in m.periodos)
            + sum(m.z[k, t] * rl[k, t] for k in m.plantas for t in m.periodos)
        )
    model.egreso_uso = Expression(rule=egreso_uso_rule)
    
    # Define egreso_transporte
    def egreso_transporte_rule(m):
        return (
            sum(
                m.q[p, i, j, t] * qa * da[i, j] * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for i in m.acopios
                for j in m.centros
                for t in m.periodos
            )
            + sum(
                m.r[p, j, k, t] * qa * dl[j, k] * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for j in m.centros
                for k in m.plantas
                for t in m.periodos
            )
            + sum(
                m.u[p, k, l, t] * qa * dp[k, l] * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for k in m.plantas
                for l in m.productores
                for t in m.periodos
                if (p, k, l, t) in m.combinations_u
            )
        )
    model.egreso_transporte = Expression(rule=egreso_transporte_rule)
    
    # Define egreso_compra
    def egreso_compra_rule(m):
        return sum(
            m.q[p, i, j, t] * qd[p] * (1 + inflation) ** ((t - 1) // 12)
            for p in m.envases
            for i in m.acopios
            for j in m.centros
            for t in m.periodos
        )
    model.egreso_compra = Expression(rule=egreso_compra_rule)
    
    # Define egreso_inspeccion
    def egreso_inspeccion_rule(m):
        return (
            sum(
                (m.r[p, j, k, t] / ta) * qc * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for j in m.centros
                for k in m.plantas
                for t in m.periodos
            )
            + sum(
                (m.u[p, k, l, t] / tl) * qc * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for k in m.plantas
                for l in m.productores
                for t in m.periodos
                if (p, k, l, t) in m.combinations_u
            )
        )
    model.egreso_inspeccion = Expression(rule=egreso_inspeccion_rule)
    
    # Define egreso_lavado
    def egreso_lavado_rule(m):
        return sum(
            (m.u[p, k, l, t] / tl) * ql * (1 + inflation) ** ((t - 1) // 12)
            for p in m.envases
            for k in m.plantas
            for l in m.productores
            for t in m.periodos
            if (p, k, l, t) in m.combinations_u
        )
    model.egreso_lavado = Expression(rule=egreso_lavado_rule)
    
    
    # Define egreso_pruebas
    def egreso_pruebas_rule(m):
        return sum(
            m.u[p, k, l, t] * qb * (1 + inflation) ** ((t - 1) // 12)
            for p in m.envases
            for k in m.plantas
            for l in m.productores
            for t in m.periodos
            if (p, k, l, t) in m.combinations_u
        )
    model.egreso_pruebas = Expression(rule=egreso_pruebas_rule)
    
    # Define egreso_trituracion
    def egreso_trituracion_rule(m):
        return (
            sum(
                m.r[p, j, k, t] * (1 - ta) / ta * qt * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for j in m.centros
                for k in m.plantas
                for t in m.periodos
            )
            + sum(
                m.u[p, k, l, t] * (1 - tl) / tl * qt * (1 + inflation) ** ((t - 1) // 12)
                for p in m.envases
                for k in m.plantas
                for l in m.productores
                for t in m.periodos
                if (p, k, l, t) in m.combinations_u
            )
        )
    model.egreso_trituracion = Expression(rule=egreso_trituracion_rule)
    
    # Define egreso_invcentros
    def egreso_invcentros_rule(m):
        return sum(
            m.ic[p, j, t] * ci[j] * (1 + inflation) ** ((t - 1) // 12)
            for p in m.envases
            for j in m.centros
            for t in m.periodos
        )
    model.egreso_invcentros = Expression(rule=egreso_invcentros_rule)
    
    # Define egreso_invplantas
    def egreso_invplantas_rule(m):
        return sum(
            m.ip[p, k, t] * cv[k] * (1 + inflation) ** ((t - 1) // 12)
            for p in m.envases
            for k in m.plantas
            for t in m.periodos
        )
    model.egreso_invplantas = Expression(rule=egreso_invplantas_rule)
    
    # Define emisiones_transporte
    def emisiones_transporte_rule(m):
        return (
            sum(
                da[i, j] * m.q[p, i, j, t]
                for p in m.envases
                for i in m.acopios
                for j in m.centros
                for t in m.periodos
            )
            + sum(
                dl[j, k] * m.r[p, j, k, t]
                for p in m.envases
                for j in m.centros
                for k in m.plantas
                for t in m.periodos
            )
            + sum(
                dp[k, l] * m.u[p, k, l, t]
                for p in m.envases
                for k in m.plantas
                for l in m.productores
                for t in m.periodos
                if (p, k, l, t) in m.combinations_u
            )
        ) * em
    model.emisiones_transporte = Expression(rule=emisiones_transporte_rule)
    
    # Define emisiones_lavado
    def emisiones_lavado_rule(m):
        return sum(
            m.u[p, k, l, t] / tl * el
            for p in m.envases
            for k in m.plantas
            for l in m.productores
            for t in m.periodos
            if (p, k, l, t) in m.combinations_u
        )
    model.emisiones_lavado = Expression(rule=emisiones_lavado_rule)
    
    # Define emisiones_trituracion
    def emisiones_trituracion_rule(m):
        return (
            sum(
                m.r[p, j, k, t] * (1 - ta) / ta * et
                for p in m.envases
                for j in m.centros
                for k in m.plantas
                for t in m.periodos
            )
            + sum(
                m.u[p, k, l, t] * (1 - tl) / tl * et
                for p in m.envases
                for k in m.plantas
                for l in m.productores
                for t in m.periodos
                if (p, k, l, t) in m.combinations_u
            )
        )
    model.emisiones_trituracion = Expression(rule=emisiones_trituracion_rule)


    def objective_rule1(model):
      return (model.ingreso_retornable + model.ingreso_triturado - model.egreso_adecuar - model.egreso_uso -\
              model.egreso_transporte - model.egreso_compra - model.egreso_inspeccion - model.egreso_lavado -\
              model.egreso_pruebas - model.egreso_trituracion - model.egreso_invcentros - model.egreso_invplantas)
    model.objective = Objective(rule=objective_rule1, sense=maximize)



    # Add constraints here following the Pyomo syntax
    # Example constraint (processing capacity at classification centers):
    def processing_capacity_centers(model, j, t):
        return sum(model.r[p, j, k, t] / ta for p in model.Envases for k in model.Plantas) <= cc[j] * model.x[j, t]

    model.ProcessingCapacityCenters = Constraint(model.Centros, model.Periodos, rule=processing_capacity_centers)

    return model
