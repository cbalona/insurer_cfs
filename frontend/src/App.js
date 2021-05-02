import "./App.css";
import axios from "axios";
import React, { useEffect, useState, useRef } from "react";
import { VictoryChart, VictoryAxis, VictoryLine, VictoryTheme, VictoryGroup, VictoryStack, VictoryBar } from "victory";

function App() {
  return <Home />;
}

function FormItem({ label, name, value, onChange }) {
  return (
    <div id={`${name}_form_item`} className="flex">
      <label className="py-1 pl-2 w-1/2 border rounded-l-md bg-gray-100 font-medium text-sm text-gray-500 capitalize">
        {label}
      </label>
      <input
        type="text"
        name={name}
        value={value}
        onChange={onChange}
        className="py-1 pr-2 w-1/2 border border-l-0 rounded-r-md text-sm text-gray-500 text-right"
      />
    </div>
  );
}

function BSChart({ bsResults, timestep, axDomain }) {
  if (bsResults === undefined || timestep === undefined) {
    return <div>Loading</div>
  }

  return (
    <VictoryChart height={300} domain={{y: axDomain.y}} domainPadding={{x:25}}>
      <VictoryStack key={1} labels={""} colorScale={["tomato", "orange", "gold"]}>
        <VictoryBar key={1} data={bsResults[0][timestep]} />
        <VictoryBar key={2} data={bsResults[1][timestep]} />
      </VictoryStack>
    </VictoryChart>
  )
}

function PLChart({ plResults, timestep, axDomain }) {
  if (plResults === undefined || timestep === undefined) {
    return <div>Loading</div>
  }

  return (
    <VictoryChart height={300} width={1000} domain={{y: axDomain.y}} domainPadding={{x:25}}>
      <VictoryBar key={1} data={plResults[timestep]} />
    </VictoryChart>
  )
}

function TimelineLineChart({ data, name, timestep, axDomain }) {
  if (data === undefined || timestep === undefined) {
    return <div>Loading</div>
  }

  return (
    <VictoryChart theme={VictoryTheme.material} width={560} height={210} domain={axDomain} >
      <VictoryAxis domainPadding={20} />
      <VictoryAxis dependentAxis />
      <VictoryLine
        data={data.filter((_, i) => i <= timestep)}
        x="index"
        y={name}
      />
    </VictoryChart>
  )
}

function Home() {
  const [projection, setProjection] = useState({});
  const [product, setProduct] = useState({
    premium: 100,
    loss_ratio: 0.7,
    term: 12,
    pattern_payment: 18,
    pattern_reporting: 6,
    comm_rate: 0.1,
  });
  const [model, setModel] = useState({ model_idx: 0 });
  const [timestep, setTimestep] = useState();
  const [axDomain, setAxDomain] = useState({ x: [0, 30], y: [-50, 100]})

  useEffect(() => {
    getProjection();
  }, [model.model_idx]);

  function getProjection() {
    axios
      .post("http://127.0.0.1:8000/", { product: product, model: model })
      .then((res) => {
        setProjection(res.data);
        console.log(res.data);
      })
      .catch((err) => {
        console.error(err);
      })
      .finally(() => {
        setTimestep(0);
        setAxDomain({...axDomain, y: [-0.5 * product.premium, product.premium]})
      });
  }

  function updateProduct(e) {
    const value = e.target.value;
    setProduct({
      ...product,
      [e.target.name]: value,
    });
  }

  function updateModel(e) {
    const value = e.target.value;
    setModel({
      ...model,
      [e.target.name]: value,
    });
  }

  function updateTimestep(e) {
    const value = e.target.value;
    console.log(projection);
    setTimestep(value);
  }


  return (
    <div class="container mx-auto my-8 py-4">
      <div class="grid grid-cols-3 gap-4">
        <FormItem
          label="Premium"
          name="premium"
          value={product.premium}
          onChange={updateProduct}
        />
        <FormItem
          label="Loss Ratio"
          name="loss_ratio"
          value={product.loss_ratio}
          onChange={updateProduct}
        />
        <FormItem
          label="Term"
          name="term"
          value={product.term}
          onChange={updateProduct}
        />
        <FormItem
          label="Pattern Payment"
          name="pattern_payment"
          value={product.pattern_payment}
          onChange={updateProduct}
        />
        <FormItem
          label="Pattern Reporting"
          name="pattern_reporting"
          value={product.pattern_reporting}
          onChange={updateProduct}
        />
        <FormItem
          label="Commission Rate"
          name="comm_rate"
          value={product.comm_rate}
          onChange={updateProduct}
        />
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="flex mx-auto my-6 justify-center">
          <select
            name="model_idx"
            id="cashflow-profile"
            className="border border-gray-500 rounded-md w-90 py-1 font-medium"
            value={model.model_idx}
            onChange={updateModel}
          >
            <option value="0">Basic cashflows, no reserves</option>
            <option value="1">UPR to spread premium earnings</option>
            <option value="2">Reserves recognise loss when reported</option>
            <option value="3">IBNR recognises loss when incurred</option>
            <option value="4">DAC to spread commission</option>
          </select>
        </div>
        <div class="flex mx-auto my-6 justify-center">
          <button
            id="post"
            className="border border-gray-500 rounded-md w-64 py-1 font-medium"
            onClick={getProjection}
          >
            Generate Cashflows
          </button>
        </div>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="border max-h-80">
          <TimelineLineChart data={projection.cashflow_results} name="cash" timestep={timestep} axDomain={axDomain} />
        </div>
        <div class="border row-span-2">
          <BSChart bsResults={projection.bs_results} timestep={timestep} axDomain={axDomain} />
        </div>
        <div class="border">
          <TimelineLineChart data={projection.profit_m_results} name="profit" timestep={timestep} axDomain={axDomain} />
        </div>
        <div class="border col-span-2 row-span-2">
          <PLChart plResults={projection.pl_results} timestep={timestep} axDomain={axDomain} />
        </div>
        <div class="border col-span-2">
          <input
            type="range"
            min="0"
            max="29"
            step="1"
            value={timestep}
            class="slider w-full"
            id="timeSlider"
            onChange={updateTimestep}
          ></input>
        </div>
        <div class="border col-span-2">
          <p>
            <div id="sliderval"></div>
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
