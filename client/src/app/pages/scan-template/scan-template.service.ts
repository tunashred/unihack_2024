import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ScanTemplateService {
  private apiUrl = 'http://127.0.0.1:5000/process_image';

  constructor(private http: HttpClient) {}

  processImage(imagePath: string, modelName: string = 'KidneyCancer'): Observable<any> {
    const url = this.apiUrl;

    // Set headers for JSON content
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    // Define the request body
    const body = {
      model_name: modelName,
      image_path: imagePath,
      output_path: '..\\models\\kidney_tumor_0001.jpg'
    };

    // Make a POST request to process the image
    return this.http.post(url, body, { headers });
  }
}
