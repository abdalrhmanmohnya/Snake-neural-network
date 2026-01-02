import React , {useState} from 'react'
import Footer from "./components/Footer/Footer";
import Header from "./components/Header/Header";
//import { wordes } from "./words";
import data from './dta.json'
import Products from './components/Products/Products';

function App() {

  const [products, setProducts] = useState (data)  


  return (
  
     <div className="layout"> 
<Header/>
  <main>
  <div className="wrapper">
     <Products products={products}/>
    <div className="filter-wrapper">Filter</div>
  </div>
  </main>
<Footer/>


     </div>
  
  );
}

export default App;
