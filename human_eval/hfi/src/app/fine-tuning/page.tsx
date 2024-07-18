import Navbar from "../components/navbar";
import useFlaskService from "../services/flask.service";
import Workspace from "./workspace";


export default function Home() {
  
  return (
    <>
    <Navbar />
    <div className="flex flex-row w-full">
        <Workspace id={1} />
    </div>
    </>
  );
}
