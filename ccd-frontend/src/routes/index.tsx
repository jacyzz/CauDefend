import { createBrowserRouter, Navigate } from 'react-router-dom';
import AppLayout from '../layout/AppLayout';
import IstHome from '../pages/ist/Index';
import InferHome from '../pages/infer/Index';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/ist" replace /> },
      { path: 'ist', element: <IstHome /> },
      { path: 'infer', element: <InferHome /> },
    ],
  },
]);


