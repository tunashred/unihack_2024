import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ScanTemplateService } from '../../pages/scan-template/scan-template.service';

@Component({
  selector: 'file-uploader',
  standalone: true,
  imports: [],
  templateUrl: './file-uploader.component.html',
  styles: ``
})
export class FileUploaderComponent {
  constructor(private route: ActivatedRoute, private scanTemplateService: ScanTemplateService) {}
  public pageId: string | null = '';
  public imageUrls: (string | ArrayBuffer)[] = [];
  public loading: boolean = false;

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.pageId = params.get('id');
    });
  }

  public saveFile(event: any) {
    const file = event.target.files[0];
    if (file && this.pageId) {
      this.loading = true; 
      this.scanTemplateService.processImage(file, this.pageId).subscribe(
        (res: Blob) => {
          const image = URL.createObjectURL(res);
          this.imageUrls.push(image);
          this.loading = false;
        },
        (error) => {
          console.error('Error processing image:', error);
          this.loading = false;
        }
      );
    }
  }
}
